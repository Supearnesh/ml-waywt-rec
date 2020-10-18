# -*- coding: utf-8 -*-
import os

from app import app
from flask import Blueprint, render_template, request, flash, redirect, url_for
from app.utils import allowed_file

blueprint = Blueprint('home', __name__)

@blueprint.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('home.index'))
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(url_for('home.index'))
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash('Image successfully uploaded')
            return render_template('upload.html', filename=filename)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(url_for('home.index'))
            

    return render_template('home/index.html')

@blueprint.route('/about')
def about():
    return render_template('home/about.html')

@blueprint.route('/results')
def results():
    return render_template('home/results.html')