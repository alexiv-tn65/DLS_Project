import os
import logging

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext

logging.basicConfig(level=logging.INFO)


# TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TOKEN = '5806405592:AAF2kMEn7hBQb0iEjZ1Mom4xnVkulXrYOJo'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

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
    await message.reply("The photos has been uploaded, I'm starting to process them.")
    pass

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