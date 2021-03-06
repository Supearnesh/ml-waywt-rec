import os

from flask import Flask, render_template
from . import settings, controllers, models
from .extensions import db
from flask_sqlalchemy import SQLAlchemy

project_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.abspath('app/static/uploads')

def create_app(config_object=settings):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)
    app.config['SQLALCHEMY_DATABASE_URI'] = settings.SQLALCHEMY_DATABASE_URI
    app.secret_key = os.urandom(16)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    create_database(app)
    register_blueprints(app)
    register_errorhandlers(app)
    return app

def create_database(app):
    db.init_app(app)

def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(controllers.home.blueprint)
    app.register_blueprint(controllers.auth.blueprint)
    return None

def register_errorhandlers(app):
    """Register error handlers."""
    @app.errorhandler(401)
    def internal_error(error):
        return render_template('401.html'), 401

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500

    return None
