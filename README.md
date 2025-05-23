# Potato_Disease_Classification

## Setup for python
1. Install Python ([Setup Instructions](https://wiki.python.org/moin/BeginnersGuide))
2. Install Python packages

```bash
pip3 install -r api/requirements.txt
```
3. Install Tensorflow Serving ([Setup Instructions](https://www.tensorflow.org/tfx/serving/setup))

## Setup for ReactJS
1. Install Nodejs ([Setup Instructions](https://nodejs.org/en/download))
2. Install NPM ([Setup Instructions](https://docs.npmjs.com/getting-started))
3. Install dependencies

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```
5. Copy `.env.example` as `.env`
6. Change API url in `.env`.
## Training the model
1. Download the data from [kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
2. Only keep folders related to Potatoes.
3. Run Jupyter Notebook in Browser.
4. Open potato_disease_training.ipynb in Jupyter Notebook.
5. In cell #3, update the path to dataset.
6. Run all the Cells one by one.
7. Copy the model generated and save it with the version number in the saved_models folder.

## Running the API
### Using FastAPI
1. Get inside `api` folder

```bash
cd api
```
2. Run the FastAPI Server using uvicorn

```bash
uvicorn main:app --reload --host 0.0.0.0
```
3. Your API is now running at `0.0.0.0:8000`

### Using FastAPI & TF Serve
1. Get inside `api` folder

```bash
cd api
```
2. Copy the `models.config` and update the paths in file
3. Run the TF Serve (Update config file path below)

```bash
docker run -t -d --name=tf_serving_potato --restart unless-stopped -p 8501:8501 -v D:\Programming\Projects\Potato_Disease:/Potato_Disease tensorflow/serving --rest_api_port=8501 --model_config_file=/Potato_Disease/models.config --allow_version_labels_for_unavailable_models
```
4. Run the FastAPI Server using uvicorn For this you can directly run it from your main.py or main_tf_serving.py using pycharm run option OR you can run it from command prompt as shown below,

```bash
uvicorn main_tf_serving:app --reload --host 0.0.0.0
```
5. Your API is now running at `0.0.0.0:8000`

## Running the frontend
1. Get inside `api` folder

```bash
cd api
```
2. Copy the `.env` and update REACT_APP_API_URL to API URL if needed.
3. Run the frontend

```bash
npm run start
```
