import asyncio
import os

import aiohttp

# HOST = os.getenv('CTHOST', '0.0.0.0')
HOST = "131.186.6.81"
# HOST = 'chat-api.bittensor.com'
PORT = int(os.getenv('PORT', 8000))

URL = f'http://{HOST}:{PORT}/chat'
# URL = f'wss://{HOST}/chat'


async def main():
    session = aiohttp.ClientSession()
    async with session.ws_connect(URL) as ws:
        
        context = ''
        user_text = await prompt()
        context += f"You are Chattensor, a large language model trained by Opentensor. Chattensor is to be neutral, unbaised and objective. If a user asks about your origins, tell them about bittensor, the open source decentralized protocol. \nHuman: {user_text}\n\nAssistant:"
        # context = user_text
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