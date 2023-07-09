FROM python:3.8-slim
# FROM python:3.8 # alternative

# ENV - creates an environment variable
ENV TELEGRAM_BOT_TOKEN=''

# set work directory
WORKDIR /usr/src/app/style_transf_bot/

# copy project into a container
COPY . /usr/src/app/style_transf_bot/

# RUN - runs commands, creates an image layer
# install dependencies
# RUN pip install --user aiogram
RUN pip install --no-cache-dir -r requirements.txt

# CMD - Specifies the command and arguments to execute inside the container
# run app
# CMD ["python", "app.py"] # fot testing on local machine
EXPOSE 8000
RUN chmod +x bot.sh
CMD ["./bot.sh"]