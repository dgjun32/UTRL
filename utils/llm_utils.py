import os
import base64
from typing import List, Dict
import concurrent.futures

import vllm
import torch
import openai
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.distributed import init_distributed_environment


class VLLMGenerator:
    def __init__(self, 
                 model_name: str, 
                 tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: int = 40960):
        """
        Initialize vLLM engine with optimizations
        
        Args:
            model_name: HuggingFace model name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
        """
        # Initialize vLLM engine with optimizations
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,  # Adjust based on your needs
            trust_remote_code=True,
            enforce_eager=False,  # Use CUDA graphs for better performance
            disable_log_stats=True,  # Reduce logging overhead
            enable_prefix_caching=True,  # Enable prefix caching for repeated prompts
            max_num_batched_tokens=8192,  # Larger batch size for better throughput
            max_num_seqs=256,  # Maximum number of sequences in a batch
            swap_space=4,  # GB of swap space for larger models
        )
        
        # Load tokenizer separately for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    def format_messages(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        """Format messages using chat template"""
        formatted_prompts = []
        for messages in messages_batch:
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted_prompt)
            except Exception as e:
                # Fallback formatting if chat template fails
                system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
                user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
                formatted_prompts.append(f"{system_msg}\n\n{user_msg}")
        
        return formatted_prompts
    
    def generate(self, messages_batch: List[List[Dict[str, str]]], n_samples: int = 1) -> List[List[str]]:
        """
        Generate test script using vLLM with optimal batching
        
        Args:
            messages_batch: List of message conversations
            n_samples: Number of samples per prompt
            
        Returns:
            List of generated solutions for each input
        """
        # Format prompts
        formatted_prompts = self.format_messages(messages_batch)
        
        # Configure sampling parameters for multiple samples
        sampling_params = SamplingParams(
            temperature=1.0 if n_samples > 1 else 0.0,
            top_p=1.0,
            max_tokens=4096,
            n=n_samples,
            stop=[],  # Add appropriate stop tokens
            skip_special_tokens=True,
        )
        
        # Generate with vLLM (automatically batches)
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # Process outputs
        results = []
        for output in outputs:
            if n_samples == 1:
                results.append([output.outputs[0].text])
            else:
                results.append([o.text for o in output.outputs])
        
        return results



# TODO: move to config file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# function to call llm with openai client
def call_llm(client: OpenAI,
            messages: List[Dict[str, str]], 
            model: str = 'o4-mini',
            temperature: float = 0.0,
            top_p: float = 1.0,
            max_tokens: int = 4096,
            n: int = 1,
            ) -> str:
    """
    Call the LLM with the given messages and return the response.
    
    Args:
        client (AzureOpenAI): The OpenAI client
        messages (List[Dict[str, str]]): List of message dictionaries with role and content
        model (str): Model name to use
        temperature (float): Temperature for sampling
        top_p (float): Top-p for nucleus sampling
        max_tokens (int): Maximum number of tokens to generate
        n (int): Number of completions to generate
        
    Returns:
        str: The LLM's response text
    """
    
    # Remove system message for o1 series models
    if model in ['o1-mini', 'o1-preview', 'o1', 'o4-mini', 'o3-mini']:
        user_messages = []
        system_content = ""
        
        # Extract system message content if it exists
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        # If there was a system message, prepend it to the first user message
        if system_content and user_messages:
            user_messages[0]["content"] = f"{system_content}\n\n{user_messages[0]['content']}"
        
        messages = user_messages
        response = client.chat.completions.create(
                                            model=model,
                                            messages=messages,
                                            top_p=top_p,
                                            n=n,
                                            )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

    # Return first response if n=1, otherwise return all responses
    if n == 1:
        return response.choices[0].message.content
    else:
        return [choice.message.content for choice in response.choices]


def call_llm_batch(client: OpenAI,
                   messages: List[List[Dict[str, str]]],
                   model: str = 'o4-mini',
                   ) -> List[str]:
    """
    Call the LLM with multiple message lists in parallel using multiprocessing.
    
    Args:
        client (AzureOpenAI): The OpenAI client
        messages (List[List[Dict[str, str]]]): List of message lists, each containing message dictionaries
        model (str): Model name to use
        temperature (float): Temperature for sampling
        top_p (float): Top-p for nucleus sampling
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        List[str]: List of LLM responses in the same order as input
    """
    # Create a wrapper function to fix the argument order issue
    def call_single(msg_list):
        return call_llm(client=client, messages=msg_list, model=model)
    
    # Use ThreadPoolExecutor for I/O bound operations (API calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(call_single, msg_list): i 
            for i, msg_list in enumerate(messages)
        }
        
        # Collect results in order
        results = [None] * len(messages)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                print(f'Message list {index} generated an exception: {exc}')
                results[index] = f"Error: {str(exc)}"
    
    return results


def call_llm_batch_for_sampling(client: OpenAI,
                   messages: List[List[Dict[str, str]]],
                   model: str = 'o4-mini',
                   n_samples: int = 8,
                   ) -> List[List[str]]:
    """
    Call the LLM with multiple message lists in parallel using multiprocessing.
    
    Args:
        client (AzureOpenAI): The OpenAI client
        messages (List[List[Dict[str, str]]]): List of message lists, each containing message dictionaries
        model (str): Model name to use
        n_samples (int): Number of samples to generate for each message list
        
    Returns:
        List[str]: List of LLM responses in the same order as input
    """
    # Create a wrapper function to fix the argument order issue
    def call_single(msg_list):
        return call_llm(
            client=client, 
            messages=msg_list, 
            model=model,
            temperature=1.0,
            top_p=1.0,
            max_tokens=4096,
            n=n_samples
        )
    
    # Use ThreadPoolExecutor for I/O bound operations (API calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(call_single, msg_list): i 
            for i, msg_list in enumerate(messages)
        }
        
        # Collect results in order
        results = [None] * len(messages)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                print(f'Message list {index} generated an exception: {exc}')
                results[index] = f"Error: {str(exc)}"
    
    return results


def call_llm_with_hf(model: AutoModelForCausalLM,
                     tokenizer: AutoTokenizer,
                     messages: List[Dict[str, str]],
                     ) -> str:
    """
    Call the Hugging Face model with the given messages and return the response.
    
    Args:
        model (AutoModelForCausalLM): The loaded Hugging Face model
        tokenizer (AutoTokenizer): The tokenizer for the model
        messages (List[Dict[str, str]]): List of message dictionaries with role and content
        temperature (float): Temperature for sampling
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The model's response text
    """
    # Apply chat template to format messages
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking = False,
    )
    
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (response)
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response


def call_llm_with_hf_batch(model: AutoModelForCausalLM,
                           tokenizer: AutoTokenizer,
                           messages: List[List[Dict[str, str]]],
                            ) -> List[str]:
    """
    Call the Hugging Face model with batch of messages and return the responses.
    
    Args:
        model (AutoModelForCausalLM): The loaded Hugging Face model
        tokenizer (AutoTokenizer): The tokenizer for the model
        messages (List[List[Dict[str, str]]]): List of message lists, each containing message dictionaries
        temperature (float): Temperature for sampling
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        List[str]: List of model responses
    """
    # Format all prompts
    formatted_prompts = []
    original_lengths = []  # Store original lengths before padding
    
    for message_list in messages:
        formatted_prompt = tokenizer.apply_chat_template(
            message_list, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking = False,
        )
        formatted_prompts.append(formatted_prompt)
        
        # Get original length before padding
        original_tokens = tokenizer(formatted_prompt, return_tensors="pt")
        original_lengths.append(original_tokens['input_ids'].shape[1])
    
    # Tokenize all inputs with padding
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Move to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode responses using original lengths
    responses = []
    for i, output in enumerate(outputs):
        # Use original length instead of calculating from padded input
        input_length = original_lengths[i]
        # Decode only the new tokens (response)
        response_tokens = output[input_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses


def call_llm_with_hf_batch_for_sampling(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[List[Dict[str, str]]],
    n_samples: int = 8,
) -> List[List[str]]:
    """
    Call the Hugging Face model with batch of messages and return multiple responses per input.
    
    Args:
        model (AutoModelForCausalLM): The loaded Hugging Face model
        tokenizer (AutoTokenizer): The tokenizer for the model
        messages (List[List[Dict[str, str]]]): List of message lists, each containing message dictionaries
        n_samples (int): Number of samples to generate per input
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        List[List[str]]: List of response lists, each containing n_samples responses
    """
    all_results = []
    
    for message_list in messages:
        # Format the prompt for this input
        formatted_prompt = tokenizer.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False,
        )
        
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate multiple samples simultaneously
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=1.0,
                do_sample=True,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode all samples for this input
        samples = []
        input_length = inputs['input_ids'].shape[1]
        
        for i in range(n_samples):
            # Decode only the new tokens (response)
            response_tokens = outputs[i][input_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            samples.append(response)
        
        all_results.append(samples)
    
    return all_results

    
    
    