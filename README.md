# MIPT DLSchool final project (advanced track) #

## Bot  Neural Style Transfer

### Installation

#### For use with docker

- Paste your bot token into the **dockerfile**:
ENV TELEGRAM_BOT_TOKEN = 'your telegram bot token'

- build the docker image:
docker build . -t my_bot_app

- run docker image:
docker run -d my_bot_app


#### Example

| foto | style foto |  result|

| <img src="https://github.com/alexiv-tn65/DLS_Project/blob/d2d46bd56c682f13aa121f8474dc24c6acde2362/examples/1/335034699_main.jpg" width="128" height="128"> | <img src="https://github.com/alexiv-tn65/DLS_Project/blob/d2d46bd56c682f13aa121f8474dc24c6acde2362/examples/1/335034699_result.jpg" width="128" height="128">   | <img src="https://github.com/alexiv-tn65/DLS_Project/blob/d2d46bd56c682f13aa121f8474dc24c6acde2362/examples/1/335034699_style.jpg" width="128" height="128">  |
