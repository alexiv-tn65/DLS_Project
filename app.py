import os
import logging
from rq import Queue
from rq.job import Job

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.types import ParseMode
import aiogram.utils.markdown as md

from models.style_transfer_model import StyleTransferNNet
from worker import connect

logging.basicConfig(level=logging.INFO)


TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

red_queue = Queue(connection=connect, default_timeout=1000)

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

device = None
st_model = StyleTransferNNet(device)

class FotoState(StatesGroup):
    photo_main = State()
    photo_style = State()
    job_id = State()


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply('Hello, I\'m a bot. I can transfer the style of one picture to another.\nTo\
        start, enter the command "/transfer_style".\nTo abort execution and start again, enter the\
        command "/abort".')

@dp.message_handler(commands=['transfer_style'])
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
    await message.reply('Canceled, you can start again using the command "/transfer_style".',
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

        red_job = red_queue.enqueue_call(
            func=st_model.run_style_transfer,
            args=(photo_main, photo_style, result_path),
            result_ttl=1000,
            ttl=1000,
            failure_ttl=1000
        )

        data['job_id'] = red_job.get_id()
        await FotoState.next()

        # st_model.run_style_transfer(photo_main, photo_style, result_path)

    await bot.send_message(
        message.chat.id,
        md.text(
            md.text('Starting to prepare photo.'),
            sep='\n',
        ),
        parse_mode=ParseMode.MARKDOWN,
    )


@dp.message_handler(state=FotoState.photo_style)
async def error_photo_style(message: types.Message):
    await message.reply("Error. I need a style photo.")



@dp.message_handler(commands='get_result', state=FotoState.job_id)
async def get_info_result(message: types.Message, state: FSMContext):
    try:
        async with state.proxy() as data:
            job = Job.fetch(data['job_id'], connection=connect)
            if job.is_finished:
                resulting_image = open(f'./photos/{message.chat.id}_result.jpg', 'rb')
                await bot.send_photo(message.chat.id, resulting_image, "Here is your photo")
                await state.finish()
            else:
                await bot.send_message(message.chat.id, "Photo not ready please wait")
    except Exception as e:
        pass


@dp.message_handler(content_types=['text'])
async def get_text_messages(message: types.Message):
   if message.text.lower() == 'Hi':
       await message.answer('Hi!')
   else:
       await message.answer('I don\'t understand what it means.')



if __name__ == '__main__':
   executor.start_polling(dp)