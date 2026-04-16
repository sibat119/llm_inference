#external imports
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#os.environ['WORLD_SIZE'] = '1'
import yaml
import torch
import os
import transformers
from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_config
from vllm.lora.request import LoRARequest


# local imports
from .chat_session import ChatSession

class VllmSession(ChatSession):
    """
    A subclass of ChatSession specifically for the LLAMA2 language model, using YAML configuration.
    """

    def __init__(self, config, model_name, temperature=0.1):
        """
        Initializes the LLAMA2 chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        
        super().__init__(config, model_name, temperature)  # Initialize the base class with the loaded configuration

        # set number of devices to use
        # num_devices = config.get(
        #     'num_devices',
        #     torch.cuda.device_count()
        # )
        
        # # self.is_generation_model = is_generation_model
        # tensor_parallel_size = self._set_tensor_parallel(num_devices)
        
        
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        
        if self.require_parallel(model_name):

            print("Using tensor GPU Memory Utilization = 0.75 with tensor parallel size = 2")        
            print(f"Attempting to load {model_name} with 4 bit precision...")
            # Quantized (4-bit)
                
            self.model = LLM(
                model=model_name,
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.85,
                max_model_len=8192,           # reduce KV cache pressure
                enforce_eager=True,           # avoids extra compile buffers
                trust_remote_code=True,
                quantization=None if "mistral" in model_name else "bitsandbytes",  # 4-bit quantization
                tokenizer_mode="mistral" if "mistral" in model_name else "auto",
                config_format="mistral" if "mistral" in model_name else "auto",
                load_format="mistral" if "mistral" in model_name else "bitsandbytes",
                distributed_executor_backend="mp" if "mistral" not in model_name else None,
                enable_lora=True,           
                max_lora_rank=16,           
            )
        else:
            self.model = LLM(
                model_name,
                trust_remote_code=True,
                enforce_eager=True,           # avoids extra compile buffers
                download_dir=config.get('model_cache', None),
                dtype=torch.bfloat16,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                tokenizer_mode="mistral" if "mistral" in model_name else "auto",
                config_format="mistral" if "mistral" in model_name else "auto",
                load_format="mistral" if "mistral" in model_name else "auto",
                enable_lora=True,           
                max_lora_rank=16,           
                # quantization="bitsandbytes",  # 4-bit quantization
                # load_format="bitsandbytes",  # use bitsandbytes format
                # max_model_len=8192,
            )
        # seqs = self.model.generate(msg,SamplingParams(top_p=self.top_p,max_tokens=self.num_output_tokens,temperature=0.01,))
        self.sampling_params = SamplingParams(
            top_p=self.top_p,
            max_tokens=self.num_output_tokens,
            temperature=self.temperature,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        # ── Compatibility shim for MistralCommonTokenizer (Mistral-Small-3.2+) ──
        # MistralCommonTokenizer does not implement the standard PreTrainedTokenizer
        # interface, so attributes like all_special_ids are missing.
        if not hasattr(self.tokenizer, 'all_special_ids'):
            self.tokenizer.all_special_ids = []
        if not hasattr(self.tokenizer, 'all_special_tokens'):
            self.tokenizer.all_special_tokens = []
        if not hasattr(self.tokenizer, 'all_special_tokens_extended'):
            self.tokenizer.all_special_tokens_extended = []
        
    def require_parallel(self, model_name):
        big_model_list = [
            'meta-llama/Llama-3.3-70B-Instruct',
            'meta-llama/Llama-4-Scout-17B-16E-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen3-30B-A3B-Instruct-2507',
            'google/gemma-3-27b-it',
            'CohereLabs/aya-expanse-32b',
            'Qwen/Qwen2.5-32B-Instruct',
            'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
            # 'google/gemma-2-9b-it',
        ]
        return model_name in big_model_list
    def get_session_type(self) -> str:
        return 'vllm'

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True,
                     temperature: int = 0.2,
                     return_logprobs: bool = False,
                     lora_dir: str = None,
                     ):
        """
        Retrieves a response from the vLLM language model.
        """
        
        sampling_params = SamplingParams(
            temperature =temperature,
            max_tokens  =self.num_output_tokens,
            top_p       =0.9,
            logprobs    = 1 if return_logprobs else None,
        )
        
        lora_request = LoRARequest("peft_adapter", 1, lora_dir) if lora_dir else None
        
        msg = self._prepare_batch_vllm(user_message, system_message)
        # Implement the logic to interact with the LLAMA2 model's API
        # This is a placeholder implementation

        # generate response
        #bsize=3
        #for i in range(0, len(msg), bsize):
        # seqs = self.model.generate(
        #     msg,
        #     sampling_params=sampling_params,
        #     use_tqdm=False,
        # )
        seqs = self.model.chat(
            msg,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
            lora_request=lora_request,
        )

        # vLLM automatically removes the prompt from the output
        # so if clean_output is set to False then we need to add the prompt back in
        texts = [seq.outputs[0].text for seq in seqs]

        if not clean_output:
            texts = [prompt + text for prompt, text in zip(msg, texts)]

        if self.model_name == "openai/gpt-oss-20b":
            answers = []
            for s in texts:
                if "So answer:" in s:
                    answers.append(s.split("So answer:")[1])
                else:
                    answers.append(s)
            texts = answers

        # ── NEW: extract logprob confidences (only when requested) ────────────
        if return_logprobs:
            logprob_confs = []
            for seq in seqs:
                token_logprobs = seq.outputs[0].logprobs   # List[Dict] or None
                conf = self._mean_exp_logprob(token_logprobs)
                logprob_confs.append(conf)
            return texts, logprob_confs   # ← (List[str], List[float])
        
        
        return texts

    @staticmethod
    def _mean_exp_logprob(token_logprobs) -> float:
        """
        Converts a vLLM token logprobs list into a scalar confidence in (0, 1].

        vLLM logprobs format (with logprobs=1):
            List[Dict[int, Logprob]] — one dict per generated token,
            each dict maps token_id → Logprob(logprob=float, ...)

        Returns mean exp(logprob) across all tokens.
        Falls back to 0.5 (neutral uncertainty) on any failure.
        """
        import math

        if not token_logprobs:
            return 0.5

        try:
            lp_values = []
            for token_dict in token_logprobs:
                if token_dict is None:
                    continue
                # Each dict has exactly one entry when logprobs=1
                top_lp = next(iter(token_dict.values())).logprob
                lp_values.append(top_lp)

            if not lp_values:
                return 0.5

            mean_lp = sum(lp_values) / len(lp_values)
            return float(math.exp(mean_lp))   # safely in (0, 1]

        except Exception:
            return 0.5
    
    def prepare_message(self, usr_msg, sys_msg):
        if self.model_name == 'Phind/Phind-CodeLlama-34B-v2':
            msg = f'### User Message\n{usr_msg}\n\n### Assistant\n'
            if sys_msg is not None:
                msg = f'### System Prompt\n{sys_msg}\n\n' + msg
            return msg
        elif 'WizardLM' in self.model_name:
            msg = f'### Instruction:\n{usr_msg}\n\n### Response:'
            if sys_msg is not None:
                msg = f'{sys_msg.strip()}\n\n' + msg
            else:
                msg = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n' + msg
            return msg
        elif 'codellama' in self.model_name and 'Instruct' in self.model_name:
            msg = f'[INST]{usr_msg.strip()}[/INST]'
            if sys_msg is not None:
                msg = f'<<SYS>>{sys_msg.strip()}<</SYS>>' + msg
            return msg
        elif 'Salesforce' in self.model_name and 'instruct' in self.model_name:
            msg = f'### Instruction:\n{usr_msg.strip()}\n\n### Response:\n'
            if sys_msg is not None:
                msg = f'{sys_msg.strip()}\n\n' + msg
            else:
                msg = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n' + msg

            return msg
        elif self.model_name == 'mistralai/Mistral-7B-Instruct-v0.1':
            msg = f'[INST] {usr_msg} [/INST]'
            return msg 
        elif 'lmsys/vicuna' in self.model_name:
            msg = f'USER: {usr_msg.strip()}\nASSISTANT: '
            if sys_msg is not None:
                msg = sys_msg.strip() + '\n\n' + msg
            return msg
        elif (self.model_name == 'meta-llama/Llama-3.2-1B-Instruct'
                or self.model_name == 'meta-llama/Llama-3.2-3B-Instruct'
                or self.model_name == 'meta-llama/Llama-3.1-8B-Instruct'
                or self.model_name == 'meta-llama/Llama-3.3-70B-Instruct'
                or self.model_name == 'microsoft/phi-4'
                or self.model_name == 'microsoft/Phi-4-mini-instruct'
                or self.model_name == 'Qwen/Qwen3-30B-A3B-Instruct-2507'
                or self.model_name == 'Qwen/Qwen2.5-7B-Instruct'
                or self.model_name == 'Qwen/Qwen2.5-14B-Instruct'
                or self.model_name == 'Qwen/Qwen2.5-32B-Instruct'
                or self.model_name == 'Qwen/Qwen2.5-72B-Instruct'
                or self.model_name == 'CohereLabs/aya-expanse-8b'
                or self.model_name == 'CohereLabs/aya-expanse-32b'
                or self.model_name == 'openai/gpt-oss-20b'
                or self.model_name == 'google/gemma-2-27b-it'
                or self.model_name == 'mistralai/Mistral-7B-Instruct-v0.3'
                or self.model_name == 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'
                or self.model_name == 'mistralai/Mixtral-8x7B-Instruct-v0.1'
                or self.model_name == '01-ai/Yi-1.5-6B-Chat'
                or self.model_name == '01-ai/Yi-1.5-34B-Chat'
                or self.model_name == "microsoft/Phi-4-mini-instruct"
                or self.model_name == "microsoft/Phi-4-mini-instruct"
                or self.model_name == "microsoft/Phi-3.5-mini-instruct"
                or self.model_name == "microsoft/phi-4"
                or self.model_name == "deepseek-ai/DeepSeek-V2-Lite-Chat"
                or self.model_name =="google/gemma-3-4b-it"
                or self.model_name == "google/gemma-3-12b-it"
                or self.model_name == "google/gemma-2-9b-it"
                or self.model_name == "tiiuae/Falcon3-7B-Instruct"
                or self.model_name == "stabilityai/stablelm-2-1_6b"
                or self.model_name == "Qwen/Qwen2.5-3B-Instruct"
                or self.model_name == "Qwen/Qwen3-8B"
                        ):
            if sys_msg:
                msg = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": usr_msg},
                    ]
            else:
                msg = [
                        {"role": "user", "content": usr_msg},
                    ]
                
            return msg
        
    def _prepare_batch_vllm(self, user_messages, system_messages):
        messages = []
        messages = [self.prepare_message(prompt, sys) 
               for prompt, sys in zip(user_messages, system_messages)]
        
        return messages
        
        
    def _set_tensor_parallel(self, num_devices):

        # get number of attention heads for the model
        n_head = self._get_num_attn_heads()

        tensor_parallel_size = num_devices
        while n_head%tensor_parallel_size != 0:
            tensor_parallel_size -= 1

        return tensor_parallel_size

    def _get_num_attn_heads(self):
        
        llm_cfg = get_config(self.model_name, trust_remote_code=True)

        # run through possible names for the number of attention heads in llm_cfg
        # this is necessary because the configs for each LLM are not standardized
        n_head = getattr(llm_cfg, 'num_attention_heads', None)
        n_head = getattr(llm_cfg, 'n_head', n_head)
        n_head = getattr(llm_cfg, 'num_heads', n_head)
        n_head = getattr(llm_cfg, 'num_attention_heads', n_head)
        try:
            n_head = getattr(getattr(llm_cfg, 'text_config', {}), "num_attention_heads", n_head)
        except:
            print("no attr found text_config")

        if n_head is None:
            print('n_head not set')
            breakpoint()
            raise ValueError()

        return n_head
