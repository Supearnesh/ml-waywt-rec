# -*- coding: utf-8 -*-
import os
import random

from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from app.utils import allowed_file

blueprint = Blueprint('home', __name__)

@blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files:
            flash('No file part')
            return redirect(url_for('home.index'))
        
        num = random.randint(1,3)
        file = request.files[f'file{num}']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(url_for('home.index'))
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash('You should wear this!')
            return render_template('home/index.html', filename=filename)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(url_for('home.index'))
            

    return render_template('home/index.html')

@blueprint.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename, code=301))

@blueprint.route('/about')
def about():
    return render_template('home/about.html')

@blueprint.route('/results')
def results():
    return render_template('home/results.html')