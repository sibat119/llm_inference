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
        num_devices = config.get(
            'num_devices',
            torch.cuda.device_count()
        )
        
        # self.is_generation_model = is_generation_model
        # tensor_parallel_size = self._set_tensor_parallel(num_devices)
        
        
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        
        if any(x in model_name.lower() for x in ['70b', '72b', '80b', '17b-16e', 'kimi-k2']):

            print("Using tensor GPU Memory Utilization = 0.75 with tensor parallel size = 2")        
            print(f"Attempting to load {model_name} with 4 bit precision...")
            # Quantized (4-bit)
                
            self.model = LLM(
                model=model_name,
                tensor_parallel_size=2,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.75,
                max_model_len=8192,           # reduce KV cache pressure
                enforce_eager=True,           # avoids extra compile buffers
                trust_remote_code=True,
                quantization="bitsandbytes",  # 4-bit quantization
                load_format="bitsandbytes",  # use bitsandbytes format
                distributed_executor_backend="mp",
            )
        else:
            self.model = LLM(
                model_name,
                trust_remote_code=True,
                enforce_eager=True,           # avoids extra compile buffers
                download_dir=config.get('model_cache', None),
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.75,
                tokenizer_mode= "mistral" if "mistral" in model_name else "auto",
                # quantization="bitsandbytes",  # 4-bit quantization
                # load_format="bitsandbytes",  # use bitsandbytes format
                max_model_len=self.max_length,
            )
        # seqs = self.model.generate(msg,SamplingParams(top_p=self.top_p,max_tokens=self.num_output_tokens,temperature=0.01,))
        self.sampling_params = SamplingParams(
            top_p=self.top_p,
            max_tokens=self.num_output_tokens,
            temperature=self.temperature,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        
    def get_session_type(self) -> str:
        return 'vllm'

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list = None,
                     clean_output:   bool = True,
                     temperature: int = 0.2):
        """
        Retrieves a response from the vLLM language model.
        """
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.num_output_tokens,
            top_p=0.9,
        )
        
        msg, return_str = self._prepare_batch(user_message, system_message)
        # Implement the logic to interact with the LLAMA2 model's API
        # This is a placeholder implementation

        # generate response
        #bsize=3
        #for i in range(0, len(msg), bsize):
        seqs = self.model.generate(
            msg,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # vLLM automatically removes the prompt from the output
        # so if clean_output is set to False then we need to add the prompt back in
        seqs = [seq.outputs[0].text for seq in seqs]
        if not clean_output:
            seqs = [prompt + seq for prompt, seq in zip(msg, seqs)]


        # Update history and usage statistics
        # [Rest of the method should handle response parsing and updating the session similar to OpenAISession]
        if return_str:
            return seqs[0]#.outputs[0].text
        else:
            #response = [seq.outputs[0].text for seq in seqs]
            return seqs

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

        if n_head is None:
            print('n_head not set')
            breakpoint()
            raise ValueError()

        return n_head
