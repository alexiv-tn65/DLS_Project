from aiogram import Bot, Dispatcher, executor, types

# API_TOKEN = '5602787567:AAGYv7NrSjwyW7qPs_yvu70C060zrcfZDbQ' #В одинарных кавычках размещаем токен, полученный от @BotFather.
API_TOKEN = '5806405592:AAF2kMEn7hBQb0iEjZ1Mom4xnVkulXrYOJo' #В одинарных кавычках размещаем токен, полученный от @BotFather.
# API_TOKEN = os.getenv('BOT_API_TOKEN')


bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start']) #Явно указываем в декораторе, на какую команду реагируем. 
async def send_welcome(message: types.Message):
   await message.reply("Привет!\nЯ Эхо-бот") #Так как код работает асинхронно, то обязательно пишем await.



@dp.message_handler() #Создаём новое событие, которое запускается в ответ на любой текст, введённый пользователем.
async def echo(message: types.Message): #Создаём функцию с простой задачей — отправить обратно тот же текст, что ввёл пользователь.
   await message.answer(message.text)



if __name__ == '__main__':
   executor.start_polling(dp, skip_updates=True)