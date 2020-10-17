"""Settings configuration - Configuration for environment variables can go in here."""

import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv('FLASK_ENV', default='production')
DEBUG = ENV == 'development'
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
SECRET_KEY = os.getenv('SECRET_KEY', default='octocat')
GITHUB_CLIENT_ID = os.getenv('55375a0e447dfdc0a601')
GITHUB_CLIENT_SECRET = os.getenv('303849f804d6b8fd2dc037553cdba2462b7d5179')
SQLALCHEMY_TRACK_MODIFICATIONS = False
