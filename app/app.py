from flask import Flask, render_template, request, jsonify
from load_data import DataModel

app = Flask(__name__)
data_model = DataModel()

@app.route('/')
def index():
    current_fires = data_model.get_top_fires()
    return render_template('home.html', current_fires=current_fires)

@app.route("/predict_single", methods=["POST"])
def translate():
    data = request.json
    #noun, adjective = str(data['noun']), str(data['adjective'])
    #print (noun, adjective)
    #noun_s, adjective_s = _translate_spanish(noun, adjective)
    #print (noun_s, adjective_s)
    #return jsonify({'noun_s': noun_s, 'adjective_s': adjective_s})
    return jsonify(data)

#def _translate_spanish(noun, adjective):
    #noun_s = noun + 'o'
    #adjective_s = adjective + 'o'
    #return noun_s, adjective_s

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)