import axios from 'axios';

const API_URL = "http://127.0.0.1:8000";

export async function getPrediction(data) {
  try {
    const response = await axios.post(`${API_URL}/predict`, data);
    return response.data;
  } catch (error) {
    console.error("Error getting prediction:", error);
    return null;
  }
}