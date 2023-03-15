from transformers import LLaMAForCausalLM, LLaMATokenizer


import torch

import asyncio
import os
from torch import nn

import aiohttp.web

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))

'''
TRANSFORMERS
'''

def load_model(model_name):
    model = LLaMAForCausalLM.from_pretrained(
                                                    model_name,
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    device_map="auto"
                                                    )

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    # model.decoder.project_in = lambda x: x.requires_grad_(True)

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    return model


def generate(prompt, max_new_tokens=1, do_sample=True, top_k=4, top_p=0.92, num_return_sequences=1, pad_token_id=50256):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    output_token_id = model.generate(
                                    input_ids, 
                                    max_new_tokens=max_new_tokens,
                                    do_sample=do_sample, 
                                    top_k=top_k, 
                                    top_p=top_p,
                                    temperature=0.5,
                                    repetition_penalty=1.03,
                                    use_cache=True,
                                    num_return_sequences=num_return_sequences, 
                                    pad_token_id=pad_token_id
                                )[0][-1].item()

    output_token = tokenizer.decode([output_token_id])
    output_token = output_token.replace(prompt, "")
    return output_token


def format_text(messages):
    formatted_text = ""
    for message in messages:
        
        

        import code; code.interact(local=dict(globals(), **locals()))
        if message[0]['speaker'] == 'human':
            formatted_text += f"<question>{message['content']}<answer>"
        else:
            formatted_text += f" {message['content']}"
    return formatted_text

'''
ASYNCIO 
'''

async def testhandle(request):
    return aiohttp.web.Response(text='Test handle')


async def websocket_handler(request):
    print('Websocket connection starting')
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection ready')

    async for msg in ws:
        print(msg)
        text = msg.data
        if msg.type == aiohttp.WSMsgType.TEXT:
            print(text)
            if msg.data == 'close':
                await ws.close()
            else:
                # await ws.send_str(msg.data + '/answer')
                running = True
                input_text = ""
                input_text = text
                print(input_text)
                # f"Human: {text}\n\nAssistant:" 
                generated_text = ""
                end_tokens = [ "<|endoftext|>", "\n\n", "Human:", "Assistant:"]
                last_three_tokens = []
                while running:
                    
                    # get the current token count
                    token_count = len(tokenizer.encode(input_text))
                    
                    # generate a new token
                    output_token = generate(input_text)

                    # add the new token to the last_three_tokens list
                    last_three_tokens.append(output_token)

                    # if last_three_tokens is longer than 3, remove the first item
                    if len(last_three_tokens) >= 2:
                        last_three_tokens.pop(0)
                    
                    # turn last_three_tokens into a string
                    last_three_tokens_str = "".join(last_three_tokens)

                    # check if token_count is greater than 2047
                    if token_count >= 2047:
                        running = False

                    # check if any end tokens are in generated_text
                    for end_token in end_tokens:
                        if end_token in generated_text:
                            await ws.send_str('<|endoftext|>')
                            running = False
                    
                    # check if it is done generating by checking if the last three tokens are end tokens
                    if last_three_tokens_str in end_tokens:
                        await ws.send_str('<|endoftext|>')
                        running = False
                    


                    # print the output_token
                    print(bytes( output_token, 'utf-8' ).decode( 'utf-8', 'ignore' ).replace("�", ""), end="", flush=True)


                    # add the output_token to the input_text
                    input_text += output_token
                    generated_text += output_token

                    # send the output_token to the client
                    if output_token not in end_tokens:
                        await ws.send_str(output_token)


    print('Websocket connection closed')
    return ws


'''
MAIN
'''

def main():
    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.get('/', testhandle),
                    aiohttp.web.get('/chat', websocket_handler)])

    aiohttp.web.run_app(app, host=HOST, port=PORT)

if __name__ == '__main__':
    # model = load_model('robertmyers/bpt-sft')
    model = load_model('/home/ubuntu/minimal-llama/models/bpt-instruct')
    # model = load_model('./bpt-sft-base')
    tokenizer = LLaMATokenizer.from_pretrained('/home/ubuntu/openRLHF/models/hf_7B/tokenizer')

    main()