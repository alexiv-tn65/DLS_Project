from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import os


TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply('Я бот. Приятно познакомиться')



@dp.message_handler(content_types=['text'])
async def get_text_messages(message: types.Message):
   if message.text.lower() == 'привет':
       await message.answer('Привет!')
   else:
       await message.answer('Не понимаю, что это значит.')



if __name__ == '__main__':
   executor.start_polling(dp)