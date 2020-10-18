# -*- coding: utf-8 -*-
import functools, json, requests

from flask import flash, redirect, render_template, request
from flask import Blueprint, session, url_for, g

blueprint = Blueprint('auth', __name__, url_prefix='/auth')

@blueprint.route('/signup', methods=['POST'])
def signup():
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form(['password']))

    # TODO

@blueprint.route('/login', methods=['POST'])
def login():
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form(['password']))

    # TODO
    pass

@blueprint.route('/callback', methods=('GET', 'POST'))
def callback():

    # TODO
    pass

@blueprint.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home.index'))

@blueprint.before_app_request
def get_current_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = User.query.filter_by(id=user_id).first() # TODO: get user ID
