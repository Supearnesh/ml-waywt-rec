from app.extensions import db

class Profile(db.Model):
    __tablename__ = 'user_data'

    id = db.Column(db.Integer(), primary_key=True)
    age = db.Column(db.String(255), nullable=False)
    gender = db.Column(db.String(255), nullable=False)
    gen_pres = db.Column(db.String(255), nullable=False)
    style = db.Column(db.String(255), nullable=False)
    fav_color = db.Column(db.String(255), nullable=False)
    loc = db.Column(db.String(255), nullable=False)

    def __init__(self, age, gender, gen_pres, style, fav_color, loc):
        self.id = id
        self.age = age
        self.gender = gender
        self.gen_pres = gen_pres
        self.style = style
        self.fav_color = fav_color
        self.loc = loc

    def __repr__(self):
        return "<Profile: {}>".format(self.id)
