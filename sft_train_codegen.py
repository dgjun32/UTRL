import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import torch
import os
import json
import wandb
from datetime import datetime
from typing import List, Dict
import numpy as np
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor

from utils.parsing_utils import extract_python_code
from utils.testing_utils import run_testcase_stdio


class SafeDataCollator:
    """Custom data collator that ensures input_ids remain as Long tensors"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Use default collator but ensure dtype
        from transformers import default_data_collator
        batch = default_data_collator(features)
        
        # Ensure input_ids are Long tensors
        if 'input_ids' in batch:
            batch['input_ids'] = batch['input_ids'].long()
        if 'labels' in batch:
            batch['labels'] = batch['labels'].long()
        if 'attention_mask' in batch:
            batch['attention_mask'] = batch['attention_mask'].long()
            
        return batch

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

def validate_dataset_samples(dataset, tokenizer, max_samples_to_check=1000):
    """Validate dataset samples and find problematic ones"""
    print(f"\n🔍 Validating dataset samples...")
    
    problematic_samples = []
    valid_samples = 0
    
    for i, example in enumerate(dataset):
        if i >= max_samples_to_check:
            break
            
        try:
            # Check if example has required fields
            if 'messages' not in example:
                problematic_samples.append({
                    'index': i,
                    'error': 'Missing messages field',
                    'example': example
                })
                continue
                
            messages = example['messages']
            
            # Check if messages is a list
            if not isinstance(messages, list):
                problematic_samples.append({
                    'index': i,
                    'error': f'Messages is not a list: {type(messages)}',
                    'example': example
                })
                continue
            
            # Check if messages list is empty
            if len(messages) == 0:
                problematic_samples.append({
                    'index': i,
                    'error': 'Empty messages list',
                    'example': example
                })
                continue
                
            # Check each message structure
            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    problematic_samples.append({
                        'index': i,
                        'error': f'Message {j} is not dict: {type(msg)}',
                        'example': example
                    })
                    break
                    
                if 'role' not in msg or 'content' not in msg:
                    problematic_samples.append({
                        'index': i,
                        'error': f'Message {j} missing role/content: {msg.keys()}',
                        'example': example
                    })
                    break
                    
                # Check if content is string
                if not isinstance(msg['content'], str):
                    problematic_samples.append({
                        'index': i,
                        'error': f'Message {j} content is not string: {type(msg["content"])}',
                        'example': example
                    })
                    break
            else:
                # Try to apply chat template
                try:
                    prompt_messages = [msg for msg in messages if msg['role'] != 'assistant']
                    if prompt_messages:
                        prompt = tokenizer.apply_chat_template(
                            prompt_messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        # Try tokenization to check if it works
                        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
                        if tokens['input_ids'].shape[1] > 16384:
                            problematic_samples.append({
                                'index': i,
                                'error': f'Sequence too long: {tokens["input_ids"].shape[1]} tokens',
                                'example': example
                            })
                            continue
                            
                    valid_samples += 1
                    
                except Exception as e:
                    problematic_samples.append({
                        'index': i,
                        'error': f'Chat template/tokenization error: {str(e)}',
                        'example': example
                    })
                    continue
                    
        except Exception as e:
            problematic_samples.append({
                'index': i,
                'error': f'General error: {str(e)}',
                'example': example
            })
    
    print(f"📊 Validation Results:")
    print(f"  ✅ Valid samples: {valid_samples}")
    print(f"  ❌ Problematic samples: {len(problematic_samples)}")
    
    if problematic_samples:
        print(f"\n🚨 First 5 problematic samples:")
        for i, prob in enumerate(problematic_samples[:5]):
            print(f"  {i+1}. Index {prob['index']}: {prob['error']}")
            
        # Save detailed problematic samples to file
        import json
        with open("problematic_samples.json", "w", encoding='utf-8') as f:
            json.dump(problematic_samples, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed problematic samples saved to problematic_samples.json")
        
        # Show error type distribution
        error_types = {}
        for prob in problematic_samples:
            error_key = prob['error'].split(':')[0]
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        print(f"\n📈 Error type distribution:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    
    return problematic_samples, valid_samples

def convert_to_completion_format(dataset, tokenizer):
    """Convert conversational format to completion format for assistant_only_loss"""
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
            'gt_tests': example.get('gt_tests', ''),
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
                    tensor_parallel_size=8,  # Adjust based on your GPU setup
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
        all_gt_tests = []
        
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
            gt_tests: str = example["gt_tests"]
            
            # Extract input and reference
            input_messages = [msg for msg in messages if msg['role'] != 'assistant']
            reference = None
            for msg in messages:
                if msg['role'] == 'assistant':
                    reference = msg['content']
                    break
            
            all_gt_tests.append(gt_tests)
        
        all_predictions = predictions
        
        # Calculate custom metrics
        custom_metrics = self.calculate_custom_metrics(
            all_predictions, 
            all_gt_tests,
            )
        
        return {
            "metrics": custom_metrics,
            "predictions": all_predictions,
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
                                all_predictions: List[str], 
                                all_gt_tests: List[str],
                                ):
        """Calculate custom validation metrics"""
        metrics_list = []
        
        # 1. Discrimination reward
        for pred, gt_tests in zip(all_predictions, all_gt_tests):
            try:
                input_output = json.loads(gt_tests)
                inputs, outputs = input_output['inputs'], input_output['outputs']
                gt_tests = []
                for inp, out in zip(inputs, outputs):
                    gt_tests.append({
                        "input": inp,
                        "output": out
                    })
                solution = extract_python_code(pred)
                def is_passed(test_case):
                    return run_testcase_stdio(solution, test_case)['passed']
                
                with ThreadPoolExecutor(max_workers=min(len(gt_tests), 88)) as executor:
                    results = list(executor.map(is_passed, gt_tests))
                n_passed = sum(results)
                gt_score = n_passed / (len(gt_tests) + 1e-6)
                
                metrics = {
                    "pass_rate": gt_score,
                    "passed": int(gt_score > 0.999)
                }
            
            except Exception as e:
                metrics = {
                    "pass_rate": 0.0,
                    "passed": 0
                }
            
            metrics_list.append(metrics)
        
        # merge the metrics computed for each example 
        merged_metrics = merge_metrics_by_average(metrics_list)
        
        return merged_metrics
        
    
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
            eval_results["metrics"], 
            eval_results["predictions"], 
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
    DEBUG_MODE = False # Set to False for full training
    DEBUG_SAMPLES = 100  # Number of samples for debugging
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_parquet_dataset("data/train_trl_sft_codegen_taco.parquet")
    eval_dataset = load_parquet_dataset("data/eval_trl_sft_codegen_taco.parquet")
    # Use subset for debugging
    if DEBUG_MODE:
        print(f"🔧 DEBUG MODE: Using only {DEBUG_SAMPLES} samples")
        train_dataset = train_dataset.select(range(min(DEBUG_SAMPLES, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(DEBUG_SAMPLES//10, len(eval_dataset))))
    
    train_dataset = train_dataset.select(range(3000))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Train dataset columns: {train_dataset.column_names}")
    
    # Load tokenizer and model
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Validate datasets before processing
    print("\n" + "="*50)
    print("VALIDATING TRAIN DATASET")
    print("="*50)
    train_problems, train_valid = validate_dataset_samples(train_dataset, tokenizer, max_samples_to_check=len(train_dataset))
    
    print("\n" + "="*50)
    print("VALIDATING EVAL DATASET") 
    print("="*50)
    eval_problems, eval_valid = validate_dataset_samples(eval_dataset, tokenizer, max_samples_to_check=len(eval_dataset))
    
    # Filter out problematic samples if any found
    if train_problems:
        print(f"\n🔧 Filtering out {len(train_problems)} problematic samples from train dataset...")
        problematic_indices = {prob['index'] for prob in train_problems}
        valid_indices = [i for i in range(len(train_dataset)) if i not in problematic_indices]
        train_dataset = train_dataset.select(valid_indices)
        print(f"✅ Train dataset size after filtering: {len(train_dataset)}")
        
    if eval_problems:
        print(f"\n🔧 Filtering out {len(eval_problems)} problematic samples from eval dataset...")
        problematic_indices = {prob['index'] for prob in eval_problems}
        valid_indices = [i for i in range(len(eval_dataset)) if i not in problematic_indices]
        eval_dataset = eval_dataset.select(valid_indices)
        print(f"✅ Eval dataset size after filtering: {len(eval_dataset)}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Convert datasets to completion format for assistant_only_loss
    print("🔄 Converting datasets to completion format...")
    print(f"Original train dataset format: {train_dataset[0].keys()}")
    
    train_dataset = convert_to_completion_format(train_dataset, tokenizer)
    eval_dataset = convert_to_completion_format(eval_dataset, tokenizer)
    
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
        output_dir="ckpt/trl-qwen3-4b-sft-codegen-taco",
        run_name="qwen3-4b-sft-codegen-taco",
        
        # Training parameters
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=5e-5,
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
        completion_only_loss=True,  # Only compute loss on assistant responses (generated code)
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
        callbacks=[custom_callback],  # Add custom callback
        #data_collator=SafeDataCollator(tokenizer)  # Add safe data collator to ensure Long tensor dtypes
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