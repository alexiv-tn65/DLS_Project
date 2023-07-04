FROM python:3.8

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
CMD ["python", "app.py"]