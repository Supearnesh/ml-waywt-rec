# -*- coding: utf-8 -*-
import functools, json, requests

from flask import flash, redirect, render_template, request
from flask import Blueprint, session, url_for, g

from app.models.user import User
from app.models.profile import Profile
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

@blueprint.route('/profile', methods=['GET', 'POST'])
def submit_profile():
    age = None
    gender = None
    gen_pres = None
    style = None
    fav_color = None
    loc = None

    if request.method == 'POST':
        age = str(request.form['age'])
        gender = str(request.form['gender'])
        gen_pres = str(request.form['gen_pres'])
        style = str(request.form['style'])
        fav_color = str(request.form['fav_color'])
        loc = str(request.form['loc'])

        try:
            reg = Profile(age, gender, gen_pres, style, fav_color, loc)
            db.session.add(reg)
            db.session.commit()
            return redirect(url_for('home.index'))
        except:
            flash('something\'s amiss')


    return render_template('home/profile.html')

@blueprint.before_app_request
def get_current_user():
    user_id = session.get('username')

    if user_id is None:
        g.user = None
    else:
        g.user = User.query.filter_by(username=user_id).first()
