import os
import logging

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext

from models.style_transfer_model import VGG

logging.basicConfig(level=logging.INFO)


# TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TOKEN = ''
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st_model = VGG(device)

class FotoState(StatesGroup):
    photo_main = State()
    photo_style = State()


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply('Hello, I\'m a bot. I can transfer the style of one picture to another.\nTo\
        start, enter the command "/transfer".\nTo abort execution and start again, enter the\
        command "/abort".')

@dp.message_handler(commands=['transfer'])
async def style_transfer_begin(message: types.Message):
    await FotoState.photo_main.set()
    await bot.send_message(message.chat.id, "Please, send a photo.")


@dp.message_handler(state='*', commands='abort')
@dp.message_handler(Text(equals='abort', ignore_case=True), state='*')
async def abort_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return

    await state.finish()
    await message.reply('Canceled, you can start again using the command "/transfer".',
                        reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=FotoState.photo_main, content_types=['photo'])
async def process_photo_main(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['photo_main'] = message.photo[-1]

    await FotoState.next()
    await bot.send_message(message.chat.id, "Now send style photo.")

@dp.message_handler(state=FotoState.photo_main)
async def error_photo_main(message: types.Message):
    await message.reply("Error. I need a photo.")


@dp.message_handler(state=FotoState.photo_style, content_types=['photo'])
async def process_photo_style(message: types.Message, state: FSMContext):
    await message.reply("Photos have been uploaded, I'm starting to process them.")
    async with state.proxy() as data:
        data['photo_style'] = message.photo[-1]
        await data['photo_main'].download(f'./photos/{message.chat.id}_main.jpg')
        await data['photo_style'].download(f'./photos/{message.chat.id}_style.jpg')

        photo_main = st_model.image_loader(f'./photos/{message.chat.id}_main.jpg')
        photo_style = st_model.image_loader(f'./photos/{message.chat.id}_style.jpg')
        result_path = f'./photos/{message.chat.id}_result.jpg'

        st_model.run_style_transfer(photo_main, photo_style, result_path)


@dp.message_handler(state=FotoState.photo_style)
async def error_photo_style(message: types.Message):
    await message.reply("Error. I need a style photo.")



@dp.message_handler(content_types=['text'])
async def get_text_messages(message: types.Message):
   if message.text.lower() == 'Hi':
       await message.answer('Hi!')
   else:
       await message.answer('I don\'t understand what it means.')



if __name__ == '__main__':
   executor.start_polling(dp)