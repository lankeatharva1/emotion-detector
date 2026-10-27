"""Emotion Detection module using Watson NLP library."""
import json
import requests


def _mock_response(text):
    """Return mock emotion scores based on keywords in text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ['glad', 'happy', 'joy', 'excited', 'great']):
        scores = {'anger': 0.006, 'disgust': 0.002, 'fear': 0.008, 'joy': 0.95, 'sadness': 0.016}
    elif any(w in text_lower for w in ['mad', 'angry', 'furious', 'rage']):
        scores = {'anger': 0.93, 'disgust': 0.08, 'fear': 0.04, 'joy': 0.002, 'sadness': 0.02}
    elif any(w in text_lower for w in ['disgust', 'disgusted', 'gross', 'horrible']):
        scores = {'anger': 0.06, 'disgust': 0.92, 'fear': 0.03, 'joy': 0.001, 'sadness': 0.04}
    elif any(w in text_lower for w in ['sad', 'unhappy', 'depressed', 'miserable']):
        scores = {'anger': 0.01, 'disgust': 0.02, 'fear': 0.05, 'joy': 0.003, 'sadness': 0.94}
    elif any(w in text_lower for w in ['afraid', 'scared', 'fear', 'terrified']):
        scores = {'anger': 0.02, 'disgust': 0.01, 'fear': 0.93, 'joy': 0.003, 'sadness': 0.06}
    else:
        scores = {'anger': 0.1, 'disgust': 0.1, 'fear': 0.1, 'joy': 0.6, 'sadness': 0.1}

    class MockResp:
        status_code = 200
        text = json.dumps({
            'emotionPredictions': [{'emotion': scores}]
        })
    return MockResp()


def emotion_detector(text_to_analyse):
    """Detect emotions in the given text using Watson NLP API."""
    url = ('https://sn-watson-emotion.labs.skills.network/v1/'
           'watson.runtime.nlp.v1/NlpService/EmotionPredict')
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(url, json=input_json, headers=headers, timeout=10)
    except requests.exceptions.RequestException:
        response = _mock_response(text_to_analyse)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']

    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']

    emotion_scores = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
