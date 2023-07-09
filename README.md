# MIPT DLSchool final project (advanced track) #

## Bot  Neural Style Transfer

### Installation

#### For use with docker-compose

- docker-compose build
- docker-compose up

#### For use with docker

- Install Redis

- Paste your bot token into the **dockerfile**:
ENV TG_BOT_TOKEN = 'your telegram bot token'

- build the docker image:
docker build . -t my_bot_app

- run docker image:
docker run -d my_bot_app

#### For use without docker

- Install packages
pip install -r requirements.txt

- Install Redis

- Set environment variables: TG_BOT_TOKEN,
REDIS_ENDPOINT, REDIS_PORT

- Run worker.py and app.py
python -u worker.py
python -u app.py 

#### Example

|   foto   |   style foto   |   result   |
|----------|----------------|------------|
| <img src="https://github.com/alexiv-tn65/DLS_Project/blob/d2d46bd56c682f13aa121f8474dc24c6acde2362/examples/1/335034699_main.jpg" width="128" height="128"> | <img src="https://github.com/alexiv-tn65/DLS_Project/blob/d2d46bd56c682f13aa121f8474dc24c6acde2362/examples/1/335034699_style.jpg" width="128" height="128">   | <img src="https://github.com/alexiv-tn65/DLS_Project/blob/d2d46bd56c682f13aa121f8474dc24c6acde2362/examples/1/335034699_result.jpg" width="128" height="128">  |
| <img src="https://github.com/alexiv-tn65/DLS_Project/blob/5646c5706ef6df26768a3b372ec67b3e77690a0c/examples/2/335034699_main.jpg" width="128" height="128"> | <img src="https://github.com/alexiv-tn65/DLS_Project/blob/5646c5706ef6df26768a3b372ec67b3e77690a0c/examples/2/335034699_style.jpg" width="128" height="128">   |  <img src="https://github.com/alexiv-tn65/DLS_Project/blob/5646c5706ef6df26768a3b372ec67b3e77690a0c/examples/2/335034699_result.jpg" width="128" height="128">  |