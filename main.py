import logging


from aiogram import Bot, Dispatcher, executor, types
# Здесь мы импортируем еще два класса - StatesGroup, отвечающий за группу состояний и State, отвечающий за сами состояния.
from aiogram.dispatcher.filters.state import State, StatesGroup

# простейшим бэкендом MemoryStorage, который хранит все данные в оперативной памяти.
# Данной строкой мы импортируем класс MemoryStorage, в котором будут храниться все данные состояний всех пользователей во время 
# работы FSM. Здесь надо понимать, что объект класса MemoryStorage хранится исключительно в 
# оперативной памяти и при перезапуске бота все данные из него стираются.
from aiogram.contrib.fsm_storage.memory import MemoryStorage

# FSMContext - это класс для хранения контекста, в котором находятся пользователи при работе с машиной состояний. 
# Через него мы будем в хэндлеры передавать информацию о состояниях и получать доступ к хранилищу MemoryStorage прямо внутри хэндлеров.
from aiogram.dispatcher import FSMContext

from aiogram.dispatcher.filters import Text

import aiogram.utils.markdown as md



# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)


# bot_token = getenv("BOT_TOKEN")
# if not bot_token:
#     exit("Error: no token provided")

# bot = Bot(token=bot_token)


# API_TOKEN = '5602787567:AAGYv7NrSjwyW7qPs_yvu70C060zrcfZDbQ' #В одинарных кавычках размещаем токен, полученный от @BotFather.
API_TOKEN = '5806405592:AAF2kMEn7hBQb0iEjZ1Mom4xnVkulXrYOJo' #В одинарных кавычках размещаем токен, полученный от @BotFather.
# API_TOKEN = os.getenv('BOT_API_TOKEN')


# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)

# Чтобы начать работать с машиной состояний, для начала нам нужно инициализировать хранилище в dp (Dispatcher)
# простейшим бэкендом MemoryStorage, который хранит все данные в оперативной памяти.
# For example use simple MemoryStorage for Dispatcher.
# MemoryStorage, как я уже говорил, это некоторое хранилище информации о том, что происходит с пользователями. В каком состоянии они находятся, 
# какие данные они вводили и т.п. Через него хэндлеры могут "общаться" между собой с сохранением нужной, для реализации логики бота, информации
storage = MemoryStorage()
# dp = Dispatcher(bot)
# Диспетчер для бота
# Диспетчер — объект, занимающийся получением апдейтов от Telegram с последующим выбором хэндлера для обработки принятого апдейта.
dp = Dispatcher(bot, storage=storage)
# Отлично! Теперь мы можем записывать данные пользователя в оперативную память.


# Перед тем как создать какое-либо состояние, нам нужно создать класс, где мы поочередно опишем все states чтобы потом без проблем переключаться между ними.
# States
# Для хранения состояний необходимо создать класс, наследующийся от класса StatesGroup, 
# внутри него нужно создать переменные, присвоив им экземпляры класса State
# Cоздаем класс, наследуемый от StatesGroup, для группы состояний нашей FSM
# Далее мы создаем свой класс , наследуемый от класса StatesGroup, в котором будем хранить группу 
# наших состояний. Название класса может быть любым, но обычно принято начинать его с FSM,
class FotoState(StatesGroup):
    # Создаем экземпляры класса State, последовательно
    # перечисляя возможные состояния, в которых будет находиться
    # бот в разные моменты взаимодейтсвия с пользователем
	job_id = State()
	photo_main = State() # Will be represented in storage as 'FotoState:photo_main'
	photo_style = State() # Состояние ожидания ввода фото для стиля
    



# апдейта message: Message
# Этот хэндлер будет срабатывать на команду /start
# Хэндлер — асинхронная функция, которая получает от диспетчера/роутера очередной апдейт и обрабатывает его.
# Хэндлер на команду /start
@dp.message_handler(commands=['start']) #Явно указываем в декораторе, на какую команду реагируем. 
async def send_welcome(message: types.Message):
    # вместо bot.send_message(...) можно написать message.answer(...) или message.reply(...)
	await message.reply("Привет!\nЯ Эхо-бот") 
	#Так как код работает асинхронно, то обязательно пишем await.


# #Создаём функцию с простой задачей — отправить обратно тот же текст, что ввёл пользователь.
# @dp.message_handler() #Создаём новое событие, которое запускается в ответ на любой текст, введённый пользователем.
# async def echo(message: types.Message):
# 	# message - входящее сообщение
#     # message.text - это его текст
#     # message.chat.id - это номер его автора
# 	await message.answer(message.text)


@dp.message_handler(commands=['transfer'])
async def style_transfer_begin(message: types.Message):
	# Мы обращаемся к классу FotoState, далее к состоянию photo_main, а методом set() мы устанавливаем данное состояние.
	await FotoState.photo_main.set()
	# send_message и передали в него два обязательных параметра - айди чата, куда отправляем, и сам текст сообщения.
	await bot.send_message(message.chat.id, "Send me photo!")




# Text — фильтр текста. Работает на большинстве обработчиков

# You can use state '*' if you need to handle all states
@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
	# У объекта state есть асинхронные методы get_data() и get_state(), по которым можно 
	# получить данные пользователя внутри машины состояний, а также текущее состояние, в котором находится пользователь.
    current_state = await state.get_state()  # # текущее машинное состояние пользователя
    print(await state.get_data())
    print("current_state  ", current_state)
    if current_state is None:
        return

    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Отменено, можете начать заново, используя команду /transfer',
                        reply_markup=types.ReplyKeyboardRemove())



# ОБРАБОТКА  photo_main
@dp.message_handler(state=FotoState.photo_main, content_types=['photo'])
async def process_photo_main(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        print(" message.photo ",  message.photo)
        # print("MAIN  message.photo[-1] ", message.photo[-1])
        data['photo_main'] = message.photo[-1]

    await FotoState.next()
    await bot.send_message(message.chat.id, "Теперь пришлите фото стиля")



@dp.message_handler(state=FotoState.photo_main)
async def error_photo_main(message: types.Message):
    await message.reply("Ошибка. Мне нужно фото. ")

# -------------------


# Чтобы получить фото в aiogram, вам нужно использовать метод Message.photo у сообщения, 
# которое пришло от пользователя. Он вернет список фотографий, которые были отправлены сообщением. Каждая фотография - это объект PhotoSize

# ОБРАБОТКА  photo_style
@dp.message_handler(state=FotoState.photo_style, content_types=['photo'])
async def process_photo_style(message: types.Message, state: FSMContext):

    # Получаем список фотографий в сообщении
    # photos = message.photo



    # Место записи в стейты:
    async with state.proxy() as data:
        data['photo_style'] = message.photo[-1]
        await data['photo_main'].download(f'./photos/{message.chat.id}_main.jpg')
        await data['photo_style'].download(f'./photos/{message.chat.id}_style.jpg')

        photo_main = style_transfer_model.load_image(f'./photos/{message.chat.id}_main.jpg')
        photo_style = style_transfer_model.load_image(f'./photos/{message.chat.id}_style.jpg')
        result_path = f'./photos/{message.chat.id}_result.jpg'

        job = redis_queue.enqueue_call(
            func=style_transfer_model.run_style_transfer,
            args=(photo_main, style_photo, result_path),
            result_ttl=600,
            ttl=600,
            failure_ttl=600
        )

        data['job_id'] = job.get_id()
        await FotoState.next()

    await bot.send_message(
        message.chat.id,
        # Markdown
        md.text(
            md.text("готово Начинаю обработку ."),
            sep='\n',
        ),
        parse_mode=ParseMode.MARKDOWN,
    )

@dp.message_handler(state=FotoState.photo_style)
async def error_photo_style(message: types.Message):
    await message.reply("Мне нужно  фото стиля.")




if __name__ == '__main__':
	# Чтобы получать сообщения от серверов Telegram воспользуемся поллингом (polling. to poll - опрашивать)
	 # - постоянным опросом сервера на наличие новых обновлений.
	# Теперь делаем нашего бота доступным в сети:
	# Запуск бота
	executor.start_polling(dp, skip_updates=True)