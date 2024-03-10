from absl.app import run
import time
import json
import math
import os
from tqdm import tqdm
import random
from functools import cached_property
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import gcsfs
import tiktoken
from transformers import GenerationConfig
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG, next_rng,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, open_file
)
from lwm.llama import LLaMAConfig, FlaxLLaMAForCausalLM


FLAGS, FLAGS_DEF = define_flags_with_default(
    max_tokens_per_batch=2_000_000,
    context_lengths_min=1000,
    context_lengths_max=32000,
    n_context_length_intervals=3,
    n_rounds=3,
    seed=1234,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


class LLMPrefillTester:

    def __init__(self,
                 context_lengths_min = 1000,
                 context_lengths_max = 126000,
                 context_lengths_num_intervals = 10,
                 n_rounds = 3,
                 print_ongoing_status = True):

        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        self.context_lengths = np.repeat(self.context_lengths, n_rounds)
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.model = Sampler()

        self.enc = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)

    def compute_max_input_length(self, context_length, buffer=1024):
        block_size = self.model.block_size
        context_length += buffer
        context_length = math.ceil(context_length / block_size) * block_size
        return int(context_length)

    def run_test(self):
        contexts = []

        full_context = '\s' * 2_000_000
        full_token = self.enc.encode(full_context)

        start = time.time()
        for context_length in self.context_lengths:
            contexts = [self.enc.decode(full_token[:context_length])]

            if len(contexts) == 0:
                continue

            max_input_length = self.compute_max_input_length(context_length)
            B = FLAGS.max_tokens_per_batch / (max_input_length + self.model.block_size)
            B = int(B / self.model.data_dim) * self.model.data_dim
            if B < self.model.data_dim:
                B = self.model.data_dim
            elif B > len(contexts):
                B = int(math.ceil(len(contexts) / self.model.data_dim) * self.model.data_dim)
            if len(contexts) % B == 0:
                n_pad = 0
            else:
                n_pad = B - len(contexts) % B
            for _ in range(n_pad):
                contexts.insert(0, contexts[0])

            pbar = tqdm(total=len(contexts))
            for i in range(0, len(contexts), B):
                contexts_i = contexts[i:i + B]
                cur_start = time.time()
                outs = self.model(contexts_i, max_input_length)
                print(f'context_length: {context_length}, batch_size: {len(contexts_i)}, elapsed: {time.time() - cur_start:.2f}s')
                print('outs', outs)
                pbar.update(len(contexts_i))
            pbar.close()
        print('elapsed', time.time() - start)
        print('done')


    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Prefill Testing...")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()



class Sampler:
    def __init__(self):
        self.mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
        self.prefix_tokenizer = LLaMAConfig.get_tokenizer(
            FLAGS.tokenizer, truncation_side='left', padding_side='left'
        )
        self.tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
        self.sharded_rng = next_rng()
        self._load_model()
        print('block_size', self.block_size)

    @property
    def block_size(self):
        # return 2 * max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size)
        return max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size) * self.mesh.shape['sp']

    @property
    def data_dim(self):
        return self.mesh.shape['dp'] * self.mesh.shape['fsdp']

    def _load_model(self):
        if FLAGS.load_llama_config != '':
            llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
            updates = LLaMAConfig(**FLAGS.llama)
            llama_config.update(dict(
                remat_block=updates.remat_block,
                remat_attention=updates.remat_attention,
                remat_mlp=updates.remat_mlp,
                scan_attention=updates.scan_attention,
                scan_mlp=updates.scan_mlp,
                scan_query_chunk_size=updates.scan_query_chunk_size,
                scan_key_chunk_size=updates.scan_key_chunk_size,
                scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
                scan_layers=updates.scan_layers,
                param_scan_axis=updates.param_scan_axis,
            ))
        else:
            llama_config = LLaMAConfig(**FLAGS.llama)

        if FLAGS.update_llama_config != '':
            llama_config.update(dict(eval(FLAGS.update_llama_config)))

        llama_config.update(dict(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ))
        llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))
        self.config = llama_config

        with jax.default_device(jax.devices("cpu")[0]):
            _, self.params = StreamingCheckpointer.load_trainstate_checkpoint(
                    FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2 ** 30
            )
            self.model = FlaxLLaMAForCausalLM(
                llama_config,
                input_shape=(1, 2**11),
                seed=FLAGS.seed,
                _do_init=False,
                dtype=get_float_dtype_by_name(FLAGS.dtype),
            )
            self.model_ps = match_partition_rules(
                LLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), self.params
            )
            shard_fns, _ = make_shard_and_gather_fns(
                self.model_ps, get_float_dtype_by_name(FLAGS.dtype)
            )

            with self.mesh:
                self.params = tree_apply(shard_fns, self.params)

    @cached_property
    def _forward_generate(self):
        def fn(params, rng, batch):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
            rng_generator = JaxRNG(rng)
            output = self.model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                params=params['params'],
                prng_key=rng_generator(),
                generation_config=GenerationConfig(
                    min_new_tokens=128,
                    max_new_tokens=self.block_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.,
                    do_sample=False,
                    num_beams=1,
                    top_k=50,
                    top_p=1.0,
                )
            ).sequences[:, batch['input_ids'].shape[1]:]
            return output, rng_generator()
        return pjit(
            fn,
            in_shardings=(self.model_ps, PS(), PS()),
            out_shardings=(PS(), PS())
        )

    def __call__(self, prompts, max_input_length):
        inputs = self.prefix_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
            return_tensors='np'
        )
        print('inputs.input_ids.shape', inputs.input_ids.shape)
        batch = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        with self.mesh:
            output, self.sharded_rng = self._forward_generate(
                self.params, self.sharded_rng, batch
            )
            output = jax.device_get(output)
        print('='*80, output.shape)
        output_text = []
        for text in list(self.tokenizer.batch_decode(output, skip_special_tokens=True)):
            if self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)
        return output_text


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    ht = LLMPrefillTester(
        context_lengths_min=FLAGS.context_lengths_min,
        context_lengths_max=FLAGS.context_lengths_max,
        context_lengths_num_intervals=FLAGS.n_context_length_intervals,
        n_rounds=FLAGS.n_rounds,
    )
    ht.start_test()

if __name__ == "__main__":
    run(main)
