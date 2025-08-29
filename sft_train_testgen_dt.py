import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from trl import clone_chat_template
import torch
import os
import json
import wandb
from datetime import datetime
from typing import List, Dict
import numpy as np
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor

from utils.parsing_utils import extract_test_cases_stdio
from utils.testing_utils import run_testcase_stdio


def merge_metrics_by_average(metrics_list):
    if not metrics_list:
        return {}

    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    merged_metrics = {}
    for key in all_keys:
        values = [metrics.get(key, 0) for metrics in metrics_list if key in metrics]
        if values:
            merged_metrics[key] = sum(values) / len(values)
        else:
            merged_metrics[key] = 0.0
    
    return merged_metrics

def load_parquet_dataset(file_path):
    """Load parquet file and convert to HuggingFace Dataset"""
    df = pd.read_parquet(file_path)
    return Dataset.from_pandas(df)

def convert_to_completion_format(dataset, tokenizer):
    """Convert conversational format to completion format for completion_only_loss"""
    def convert_example(example):
        messages = example['messages']
        
        # Extract system + user messages as prompt
        prompt_messages = [msg for msg in messages if msg['role'] != 'assistant']
        
        # Extract assistant message as completion
        assistant_message = None
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_message = msg['content']
                break
        
        if assistant_message is None:
            # Skip examples without assistant response
            return None
        
        # Format prompt using chat template (without generation prompt)
        prompt = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True  # This adds the assistant token start
        )
        
        return {
            'prompt': prompt,
            'completion': assistant_message,
            # Keep original data for validation
            'messages': example['messages'],
            'extra_info': example.get('extra_info', {}),
            'candidate_solutions': example.get('candidate_solutions', []),
        }
    
    # Convert all examples
    converted = []
    for example in dataset:
        converted_example = convert_example(example)
        if converted_example is not None:
            converted.append(converted_example)
    
    return Dataset.from_list(converted)

class CustomValidationCallback(TrainerCallback):
    """Custom callback for validation with inference and custom metrics"""
    
    def __init__(self, eval_dataset, tokenizer, model_name, output_dir, eval_every_n_epochs=1):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.output_dir = output_dir
        self.eval_every_n_epochs = eval_every_n_epochs
        self.last_eval_epoch = -1
        self.vllm_engine = None
        
        # Create output directory for validation results
        self.validation_dir = os.path.join(output_dir, "validation_results")
        os.makedirs(self.validation_dir, exist_ok=True)
    '''
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Perform validation before training starts"""
        print(f"\n🔍 Starting initial validation before training...")
        
        # Perform custom evaluation
        eval_results = self.custom_evaluate(model, epoch=-1)  # epoch -1 for initial
        
        # Log to wandb
        wandb.log({
            "custom_eval_epoch": -1,
            "phase": "initial",
            **eval_results["metrics"]
        })
        
        # Save detailed results
        self.save_validation_results(eval_results, epoch=-1)
        
        print(f"✅ Initial validation completed")
    '''
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Perform custom validation at the end of each epoch"""
        current_epoch = int(state.epoch)
        
        # Check if we should evaluate this epoch
        if current_epoch - self.last_eval_epoch >= self.eval_every_n_epochs:
            print(f"\n🔍 Starting custom validation at epoch {current_epoch}...")
            
            # Perform custom evaluation
            eval_results = self.custom_evaluate(model, current_epoch)
            
            # Log to wandb
            wandb.log({
                "custom_eval_epoch": current_epoch,
                "phase": "training",
                **eval_results["metrics"]
            })
            
            # Save detailed results
            self.save_validation_results(eval_results, current_epoch)
            
            self.last_eval_epoch = current_epoch
            print(f"✅ Custom validation completed for epoch {current_epoch}")
    
    def get_vllm_engine(self, model=None, epoch=None):
        """Initialize or get vLLM engine for fast inference"""
        try:
            from vllm import LLM
            
            # If we have a current training model, save it temporarily and use it
            if model is not None:
                import tempfile
                import shutil
                
                # Create temporary directory for checkpoint
                temp_dir = tempfile.mkdtemp(prefix=f"vllm_checkpoint_epoch_{epoch}_")
                print(f"💾 Saving current model to temporary checkpoint: {temp_dir}")
                
                # Save current model state
                model.save_pretrained(temp_dir)
                self.tokenizer.save_pretrained(temp_dir)
                
                # Initialize vLLM with current checkpoint
                print(f"🚀 Initializing vLLM engine with current checkpoint...")
                vllm_engine = LLM(
                    model=temp_dir,
                    tensor_parallel_size=4,  # Adjust based on your GPU setup
                    gpu_memory_utilization=0.8,  # Leave some memory for training model
                    trust_remote_code=True,
                    max_model_len=4096,
                    dtype="bfloat16"
                )
                
                # Clean up temporary directory
                def cleanup_temp_dir():
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"🧹 Cleaned up temporary checkpoint: {temp_dir}")
                    except:
                        pass
                
                # Store cleanup function
                vllm_engine._temp_cleanup = cleanup_temp_dir
                
                print("✅ vLLM engine initialized with current model state")
                return vllm_engine
                
            # Fallback to base model (for initial validation)
            elif self.vllm_engine is None:
                print(f"🚀 Initializing vLLM engine with base model: {self.model_name}...")
                self.vllm_engine = LLM(
                    model=self.model_name,
                    tensor_parallel_size=4,
                    gpu_memory_utilization=0.8,
                    trust_remote_code=True,
                    max_model_len=4096,
                    dtype="bfloat16"
                )
                print("✅ vLLM engine initialized successfully")
                
            return self.vllm_engine
            
        except ImportError:
            print("⚠️ vLLM not available, falling back to standard inference")
            return None
        except Exception as e:
            print(f"⚠️ vLLM initialization failed: {e}, falling back to standard inference")
            return None
    
    def custom_evaluate(self, model, epoch):
        """Perform inference on validation set and calculate custom metrics"""
        all_predictions = []
        all_references = []
        all_inputs = []
        all_gt_solutions = []
        all_candidate_solutions = []
        
        print(f"Running inference on {len(self.eval_dataset)} validation examples...")
        
        # Temporarily move training model to CPU to free GPU memory for vLLM
        print("💾 Moving training model to CPU to free GPU memory...")
        original_device_map = {}
        
        # Store original device locations for parameters
        for name, param in model.named_parameters():
            original_device_map[name] = param.device
            
        # Store original device locations for buffers
        for name, buffer in model.named_buffers():
            original_device_map[f"buffer_{name}"] = buffer.device
        
        # Move model to CPU
        model.cpu()
        torch.cuda.empty_cache()
        print("✅ Training model moved to CPU, GPU memory freed")
        
        try:
            # Try to use vLLM with current model state
            vllm_engine = self.get_vllm_engine(model=model, epoch=epoch)
            
            if vllm_engine is not None:
                # Use vLLM for batch inference
                predictions = self.batch_inference_vllm(vllm_engine)
                
                # Clean up vLLM engine
                print("🧹 Cleaning up vLLM engine...")
                if hasattr(vllm_engine, '_temp_cleanup'):
                    vllm_engine._temp_cleanup()
                del vllm_engine
                torch.cuda.empty_cache()
                print("✅ vLLM engine cleaned up")
                
            else:
                # Fallback to standard inference (but model is on CPU, so move back temporarily)
                print("⚠️ vLLM failed, using standard inference...")
                model.cuda()  # Move back to GPU for inference
                model.eval()
                predictions = self.standard_inference(model)
                model.cpu()  # Move back to CPU
                
        finally:
            # Always restore training model to original GPU locations
            print("🔄 Restoring training model to original GPU locations...")
            
            # Restore each parameter to its original device
            for name, param in model.named_parameters():
                if name in original_device_map:
                    original_device = original_device_map[name]
                    param.data = param.data.to(original_device)
            
            # Also restore buffers (like batch norm stats)
            for name, buffer in model.named_buffers():
                buffer_key = f"buffer_{name}"
                if buffer_key in original_device_map:
                    original_device = original_device_map[buffer_key]
                    buffer.data = buffer.data.to(original_device)
            
            torch.cuda.empty_cache()
            print("✅ Training model restored to original device configuration")
        
        # Extract references and inputs
        for i, example in enumerate(self.eval_dataset):
            messages: List[Dict] = example['messages']
            gt_solution: str = example["extra_info"]["gt_solution"]
            candidate_solutions: List[str] = example["candidate_solutions"]
            
            # Extract input and reference
            input_messages = [msg for msg in messages if msg['role'] != 'assistant']
            reference = None
            for msg in messages:
                if msg['role'] == 'assistant':
                    reference = msg['content']
                    break
            
            all_inputs.append(input_messages)
            all_references.append(reference)
            all_gt_solutions.append(gt_solution)
            all_candidate_solutions.append(candidate_solutions)
        
        all_predictions = predictions
        
        # Calculate custom metrics
        custom_metrics, metrics_list = self.calculate_custom_metrics(
            all_predictions, 
            all_references,
            all_gt_solutions,
            all_candidate_solutions
            )
        
        return {
            "inputs": all_inputs,
            "metrics": custom_metrics,
            "metrics_list": metrics_list,
            "predictions": all_predictions,
            "references": all_references,
        }
    
    def batch_inference_vllm(self, vllm_engine):
        from vllm import SamplingParams
        
        # Prepare all prompts
        prompts = []
        for example in self.eval_dataset:
            messages = example['messages']
            input_messages = [msg for msg in messages if msg['role'] != 'assistant']
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                input_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.1,
            top_p=0.1,
            stop=["<|im_end|>"]
        )
        
        print("🚀 Running batch inference with vLLM...")
        
        # Generate in batches to manage memory
        batch_size = 128
        all_predictions = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_outputs = vllm_engine.generate(batch_prompts, sampling_params)
            
            batch_predictions = [output.outputs[0].text.strip() for output in batch_outputs]
            all_predictions.extend(batch_predictions)
            
            print(f"  Processed {min(i+batch_size, len(prompts))}/{len(prompts)} examples")
        
        return all_predictions
    
    def standard_inference(self, model):
        """Batch inference method for faster processing"""
        all_predictions = []
        batch_size = 128
        
        # Prepare all prompts first
        all_prompts = []
        for example in self.eval_dataset:
            messages = example['messages']
            input_messages = [msg for msg in messages if msg['role'] != 'assistant']
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                input_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            all_prompts.append(prompt)
        
        print(f"🚀 Running batch inference with batch size {batch_size}...")
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(all_prompts), batch_size):
                batch_prompts = all_prompts[i:i+batch_size]
                
                print(f"  Processing batch {i//batch_size + 1}/{(len(all_prompts)-1)//batch_size + 1} "
                      f"({min(i+batch_size, len(all_prompts))}/{len(all_prompts)} examples)")
                
                # Tokenize batch
                batch_inputs = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=32768
                )
                batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
                
                # Generate batch
                batch_outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode batch results
                for j, output in enumerate(batch_outputs):
                    # Get only the generated part (after input)
                    input_length = batch_inputs['input_ids'][j].shape[0]
                    generated_tokens = output[input_length:]
                    prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    all_predictions.append(prediction.strip())
        return all_predictions
    
    def generate_response(self, model, input_messages):
        """Generate response for given input messages (standard method)"""
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                input_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=32768)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return prediction.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""
    
    def calculate_custom_metrics(self, 
                                predictions: List[str], 
                                references: List[str],
                                gt_solutions: List[str],
                                candidate_solutions_list: List[List[str]]):
        """Calculate custom validation metrics"""
        metrics_list = []
        
        # 1. Discrimination reward
        for pred, ref, gt_solution, candidate_solutions in zip(predictions, references, gt_solutions, candidate_solutions_list):
            try:
                uts = extract_test_cases_stdio(pred)
                unique_uts = []
                for ut in uts:
                    if ut not in unique_uts:
                        unique_uts.append(ut)
                
                reasoning_count = pred.count("<reasoning>")
                no_degeneration = pred.strip().endswith('```')
                formatting_reward = float(reasoning_count == len(uts) and no_degeneration)
                
                if len(unique_uts) == 0:
                    discrimination_reward = 0.0
                    entire_discrimination_reward = 0.0
                    clipped_validity = 0.0
                    validity_reward = 0.0
                    brievity_penalty = 1.0
                    duplication_penalty = 0.0
                    final_score = 0.1*formatting_reward + 0.85*entire_discrimination_reward + 0.05*clipped_validity
                    
                    metrics = {
                        "score": final_score,
                        "formatting_reward": formatting_reward,
                        "n_unique_test_cases": len(unique_uts),
                        "n_test_cases": len(uts),
                        "clipped_validity": clipped_validity,
                        "validity_ratio": validity_reward,
                        "entire_discrimination_reward": entire_discrimination_reward,
                        "brievity_penalty": brievity_penalty,
                        "duplication_penalty": duplication_penalty
                    }
                
                else:
                    # compute brievity penalty
                    brievity_penalty = 1 / (len(unique_uts))
                    duplication_penalty = 1 - len(unique_uts) / len(uts)
                    # 1. GT solution 실행 - 병렬 처리
                    def test_gt_solution(ut):
                        return run_testcase_stdio(gt_solution, ut)["passed"]
                    
                    # ThreadPoolExecutor로 GT 테스트 병렬 실행
                    max_workers = min(88, mp.cpu_count(), len(unique_uts))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        gt_results = list(executor.map(test_gt_solution, unique_uts))
                    gt_results = np.array(gt_results)
                    
                    # 2. Validity reward: GT 기준 pass된 비율
                    validity_reward = gt_results.mean()
                    clipped_validity = gt_results.sum() / max(12.0, len(unique_uts))

                    # 3. Entire discrimination reward 계산: GT에서 pass된 테스트만, 나머지 candidates 평가
                    if validity_reward == 0.0:
                        entire_discrimination_reward = 0.0
                    else:
                        idx_passed = np.where(gt_results == 1)[0]
                        failed_candidate_indices = set()
                        
                        # 모든 (test_case, candidate) 쌍을 한 번에 병렬 처리
                        test_candidate_pairs = []
                        for i in idx_passed:
                            for j, candidate in enumerate(candidate_solutions):
                                test_candidate_pairs.append((i, j, unique_uts[i], candidate))
                        
                        def test_single_pair(pair):
                            test_idx, candidate_idx, ut, candidate = pair
                            try:
                                passed = run_testcase_stdio(candidate, ut)["passed"]
                                return (test_idx, candidate_idx, not passed)
                            except:
                                return (test_idx, candidate_idx, False)
                        
                        # 모든 쌍을 한 번에 병렬 처리
                        max_workers = min(88, mp.cpu_count(), len(test_candidate_pairs))
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            results = list(executor.map(test_single_pair, test_candidate_pairs))
                        
                        # 결과를 테스트 케이스별로 그룹화하여 fail한 candidate 인덱스 수집
                        for test_idx, candidate_idx, is_failed in results:
                            if is_failed:
                                failed_candidate_indices.add(candidate_idx)
                        
                        entire_discrimination_reward = len(failed_candidate_indices) / len(candidate_solutions)

                    final_score = 0.1*formatting_reward + 0.85*entire_discrimination_reward + 0.05*clipped_validity

                    metrics = {
                        "score": final_score,
                        "formatting_reward": formatting_reward,
                        "n_test_cases": len(uts),
                        "n_unique_test_cases": len(unique_uts),
                        "clipped_validity": clipped_validity,
                        "validity_ratio": validity_reward,
                        "entire_discrimination_reward": entire_discrimination_reward,
                        "brievity_penalty": brievity_penalty,
                        "duplication_penalty": duplication_penalty
                    }
            except Exception as e:
                metrics = {
                    "score": 0,
                    "formatting_reward": 0,
                    "n_unique_test_cases": 0,
                    "n_test_cases": 0,
                    "clipped_validity": 0,
                    "validity_ratio": 0,
                    "entire_discrimination_reward": 0,
                    "brievity_penalty": 0,
                    "duplication_penalty": 0,
                    }
                
            metrics_list.append(metrics)
        
        # merge the metrics computed for each example 
        merged_metrics = merge_metrics_by_average(metrics_list)
        
        return merged_metrics, metrics_list
        
    
    def save_validation_results(self, eval_results, epoch):
        """Save validation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Handle initial validation (epoch -1)
        epoch_str = "initial" if epoch == -1 else str(epoch)
        
        # Save detailed results
        results_file = os.path.join(
            self.validation_dir, 
            f"validation_epoch_{epoch_str}.json"
        )
        
        # Prepare data for saving
        detailed_results = []
        for i, (metric, pred) in enumerate(zip(
            eval_results["metrics_list"], 
            eval_results["predictions"]
        )):
            detailed_results.append({
                "example_id": i,
                "prediction": pred,
                "metric": metric
            })
        
        save_data = {
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": eval_results["metrics"],
            "results": detailed_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Also save a simple predictions file
        predictions_file = os.path.join(
            self.validation_dir,
            f"predictions_epoch_{epoch_str}.json"
        )
        
        with open(predictions_file, 'w') as f:
            json.dump(eval_results["predictions"], f)
        
        print(f"💾 Validation results saved:")
        print(f"  Detailed: {results_file}")
        print(f"  Predictions: {predictions_file}")

def main():
    # Model configuration
    model_name = "Qwen/Qwen3-4B"
    
    # Debugging options
    DEBUG_MODE = True # Set to False for full training
    DEBUG_SAMPLES = 100  # Number of samples for debugging
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_parquet_dataset("data/train_trl_gemini_distill_taco.parquet")
    eval_dataset = load_parquet_dataset("data/eval_trl_gemini_distill_taco.parquet")
    
    # Use subset for debugging
    if DEBUG_MODE:
        print(f"🔧 DEBUG MODE: Using only {DEBUG_SAMPLES} samples")
        train_dataset = train_dataset.select(range(min(DEBUG_SAMPLES, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(DEBUG_SAMPLES//10, len(eval_dataset))))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Train dataset columns: {train_dataset.column_names}")
    
    # Load tokenizer and model
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Convert datasets to completion format for completion_only_loss
    print("🔄 Converting datasets to completion format...")
    print(f"Original train dataset format: {train_dataset[0].keys()}")
    
    train_dataset = convert_to_completion_format(train_dataset, tokenizer)
    eval_dataset = convert_to_completion_format(eval_dataset, tokenizer)
    breakpoint()
    print(f"✅ Converted to completion format!")
    print(f"New train dataset format: {train_dataset[0].keys()}")
    print(f"Train dataset size after conversion: {len(train_dataset)}")
    print(f"Eval dataset size after conversion: {len(eval_dataset)}")
    
    # Show example of converted data
    example = train_dataset[0]
    print(f"\n📝 Example conversion:")
    print(f"Prompt: {example['prompt'][:200]}...")
    print(f"Completion: {example['completion'][:100]}...")
    
    # Training configuration
    training_args = SFTConfig(
        # Output and logging
        output_dir="./ckpt/qwen3-4b-sft-taco-distill-gemini-2.5-flash",
        run_name="qwen3-4b-sft-taco-distill-gemini-2.5-flash",
        
        # Training parameters
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        
        # Evaluation and saving
        #eval_strategy="epoch",
        #eval_steps=1,
        save_strategy="no",
        #save_steps=1,
        #save_total_limit=10,
        #load_best_model_at_end=True,
        #metric_for_best_model="score",
        #greater_is_better=True,
        
        # Logging
        logging_dir="./logs",
        logging_steps=1,
        report_to="wandb",
        
        # Memory optimization
        fp16=False,
        bf16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        # SFT specific parameters
        max_length=16384,
        completion_only_loss=True,  # Only compute loss on completion part (generated tests)
        packing=False,  # Set to True if you want to pack sequences for efficiency
        
        # EOS token for Qwen models
        eos_token="<|im_end|>",
        
        # Other parameters
        remove_unused_columns=False,
        seed=42,
        
        # CPU optimization for faster data preprocessing
        dataset_num_proc=32,
    )
    
    # Create custom validation callback
    custom_callback = CustomValidationCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model_name=model_name,  # Pass model name for vLLM
        output_dir=training_args.output_dir,
        eval_every_n_epochs=1  # Evaluate every epoch
    )
    
    # Initialize trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[custom_callback]  # Add custom callback
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 