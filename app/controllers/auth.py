# -*- coding: utf-8 -*-
import functools, json, requests

from flask import flash, redirect, render_template, request
from flask import Blueprint, session, url_for, g

from app.models.user import User
from app.extensions import db

blueprint = Blueprint('auth', __name__)

@blueprint.route('/signup', methods=['GET', 'POST'])
def signup():
    username = None
    email = None
    password = None

    if request.method == 'POST':
        username = str(request.form['username'])
        email = str(request.form['email'])
        password = str(request.form['password'])

        try:
            if not db.session.query(User).filter(User.email == email).count():
                reg = User(username, email, password)
                db.session.add(reg)
                db.session.commit()
                return redirect(url_for('home.index'))
            else:
                flash('email already in use')
        except:
            pass


    return render_template('home/signup.html')

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    username = None
    password = None

    if request.method == 'POST':
        username = str(request.form['username'])
        password = str(request.form['password'])

        query = db.session.query(User).filter(User.username.in_([username]), User.password.in_([password]) )
        result = query.first()
        print(result)
        if result:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home.index'))
        else:
            flash('wrong username or password!')
    return render_template('home/login.html')

@blueprint.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home.index'))

@blueprint.before_app_request
def get_current_user():
    user_id = session.get('username')

    if user_id is None:
        g.user = None
    else:
        g.user = User.query.filter_by(username=user_id).first()
