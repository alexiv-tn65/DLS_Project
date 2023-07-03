FROM python:3.8

# ENV - creates an environment variable
ENV TELEGRAM_BOT_TOKEN=''

# set work directory
WORKDIR /usr/src/app/

# copy project into a container
COPY . /usr/src/app/

# RUN - runs commands, creates an image layer
# install dependencies
RUN pip install --user aiogram

# CMD - Specifies the command and arguments to execute inside the container
# run app
CMD ["python", "app.py"]