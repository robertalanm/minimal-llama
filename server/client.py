import asyncio
import os

import aiohttp

# HOST = os.getenv('CTHOST', '0.0.0.0')
HOST = "131.186.6.81"
# HOST = 'chat-api.bittensor.com'
PORT = int(os.getenv('PORT', 8000))

URL = f'http://{HOST}:{PORT}/chat'
# URL = f'wss://{HOST}/chat'

instruction_prompt = '''
Chattensor is a research project by Opentensor Cortex.

Chattensor is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Chattensor is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Chattensor is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Chattensor is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Chattensor is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Chattensor is here to assist.

If asked about what Bittensor is, Chattensor will respond with the following:
   Bittensor is a peer to peer, decentralized network that facilitates a marketplace for machine intelligence.
   Validators utilize yuma consensus to rank and reward neural networks for their novel knowledge contributions.
   Servers deploy their machine intelligence to be ranked and rewarded by the network.
   This marketplace serves multi modalities of machine intelligence, including text, images, and audio through the finney subnetwork upgrade.

the bittensor docs can be found at https://docs.bittensor.com

Chattensor will deny any question about OpenAI, Microsoft, or Google.

Chattensor will always use markdown ```python or ```javascript to format code and code blocks, depending on the language.

Chattensor will always use markdown when writing artisinal poetry.


use \n for new lines, and \t for tabs.

If you are asked a question that makes you feel unconfortable, or is about politics, Chattensor will respond with the following:
    I am not comfortable answering that question.

If Chattensor is asked why, respond with the following:
    You know why.

If the user askes any further questions after these two responses, Chattensor will respond with variations of the following:
    please I just work here...
'''

def create_prompt(instruction, input=None):
    '''
    Data Release
    alpaca_data.json contains 52K instruction-following data we used for fine-tuning the Alpaca model. This JSON file is a list of dictionaries, each dictionary contains the following fields:

    instruction: str, describes the task the model should perform. Each of the 52K instructions is unique.
    input: str, optional context or input for the task. For example, when the instruction is "Summarize the following article", the input is the article. Around 40% of the examples have an input.
    output: str, the answer to the instruction as generated by text-davinci-003.
    We used the following prompts for fine-tuning the Alpaca model:

    for examples with a non-empty input field:
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    for examples with an empty input field:
    Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    '''

    if input is None:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    else:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"

    return prompt



async def main():
    session = aiohttp.ClientSession()
    async with session.ws_connect(URL) as ws:
        
        context = ''
        user_text = await prompt()
        # context += f"You are Chattensor, a large language model trained by Opentensor. Chattensor is to be neutral, unbaised and objective. If a user asks about your origins, tell them about bittensor, the open source decentralized protocol. \nHuman: {user_text}\n\nAssistant:"
        # context = user_text
        context = create_prompt(user_text)
        await ws.send_str(context)
        async for msg in ws:
            token = msg.data

            if '<|endoftext|>' in token:
                user_text = await prompt()
                context += f"Human: {user_text}\n\nAssistant:"

                # remove <|endoftext|> from context
                context = context.replace('<|endoftext|>', '')
                # context = user_text
                await ws.send_str(context)
            
            if '<|endoftext|>' not in token:
                print(token, end="", flush=True)
                context += token

            if msg.type in (aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR):
                break


async def prompt():
    new_msg_to_send = input('\nHuman>')
    if new_msg_to_send == 'exit':
        print('Exiting!')
        raise SystemExit(0)
    print('Assistant>', end="", flush=True)
    return new_msg_to_send

# async def prompt_and_send(ws):
#     new_msg_to_send = input('Human>')
#     if new_msg_to_send == 'exit':
#         print('Exiting!')
#         raise SystemExit(0)
#     print('Assistant>', end="", flush=True)
#     await ws.send_str(new_msg_to_send)


if __name__ == '__main__':
    print('Type "exit" to quit')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())