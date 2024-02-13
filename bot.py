from aiomultiprocess import Pool
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
import asyncio
from PIL import Image
from io import BytesIO
import time
import os
import subprocess
import re
import aiohttp
import functools
import concurrent.futures

API_TOKEN = ''

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

SAVE_PATH = 'D:/yolo/yolotestnero/photobot'

# Create a lock to ensure sequential processing for the same user
user_locks = {}


async def clean_result(result):
    cleaned_result = result.replace("192x192", "").strip()
    return cleaned_result


async def analyze_image(image_path, session):
    try:
        async with session:
            command = f"yolo task=classify mode=predict model=runs/classify/train/weights/best.pt source={image_path}"
            process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()

            output_text = stdout.decode()
            start_index = output_text.find("image 1/1")

            if start_index != -1:
                result_text = output_text[start_index:]
                results = re.findall(r"(\w+ \w+ \w+ \d+\.\d+)", result_text)
                max_result = max(results, key=lambda x: float(x.split()[-1]))
                cleaned_result = await clean_result(max_result)
                for line in results:
                    print(line.strip())
                print("Самая большая вероятность" + " " + cleaned_result)
            else:
                print("Результаты не найдены.")

            return cleaned_result
    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return None


async def send_message_with_results(user_id, image, start_time, analysis_result):
    try:
        resized_image = image.resize((32, 32))
        output_buffer = BytesIO()
        resized_image.save(output_buffer, format='JPEG')
        output_buffer.seek(0)
        response = await bot.send_photo(user_id, output_buffer)

        uploaded_file_path = (await bot.get_file(response.photo[-1].file_id)).file_path
        uploaded_image_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{uploaded_file_path}"

        end_time = time.time()

        await bot.send_message(user_id, f"Фотография получена: {uploaded_image_url}\n"
                                        f"Результат анализа: {analysis_result}\n"
                                        f"Время выполнения: {end_time - start_time:.2f} сек.")
    except Exception as e:
        print(f"An error occurred during sending a message with results: {e}")


async def process_image(user_id, photo, file_info, session):
    try:
        user_folder_path = os.path.join(SAVE_PATH, str(user_id))
        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)

        file_id = photo.file_id
        file_path = file_info.file_path

        start_time = time.time()
        print(f"Processing image from user {user_id}")

        image_file = await bot.download_file(file_path)
        image = Image.open(BytesIO(image_file.read()))

        file_name = f'{user_id}.jpg'
        file_path = os.path.join(user_folder_path, file_name)
        image.save(file_path)

        analyze_partial = functools.partial(analyze_image, session=session)
        result = await analyze_partial(file_path)

        if result is not None:
            await send_message_with_results(user_id, image, start_time, result)

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
    finally:
        # Release the lock after processing an image
        user_locks[user_id].release()


@dp.message_handler(commands=['help'])
async def handle_help(message: types.Message):
    help_text = "Для получения информации о растении, отправьте фото растения."
    await message.reply(help_text)


@dp.message_handler(commands=['id'])
async def handle_id(message: types.Message):
    user_id = message.from_user.id
    await message.reply(f"Ваш Telegram ID: {user_id}")


@dp.message_handler(content_types=ContentType.PHOTO)
async def handle_photos(message: types.Message):
    try:
        user_id = message.from_user.id

        # Acquire the lock for the user
        if user_id not in user_locks:
            user_locks[user_id] = asyncio.Lock()
        await user_locks[user_id].acquire()

        session = aiohttp.ClientSession()  # Create a session
        await process_image(user_id, message.photo[-1], await bot.get_file(message.photo[-1].file_id), session)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await session.close()  # Close the session


if __name__ == '__main__':
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    loop = asyncio.get_event_loop()
    from aiogram import executor

    executor.start_polling(dp, on_startup=None)
