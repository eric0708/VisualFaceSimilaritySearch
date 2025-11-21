import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const fetchModels = async () => {
  const response = await axios.get(`${API_BASE_URL}/models`);
  return response.data.models;
};

export const fetchSamples = async () => {
  const response = await axios.get(`${API_BASE_URL}/samples`);
  return response.data.images;
};

export const loadModel = async (modelName) => {
  const formData = new FormData();
  formData.append('model', modelName);
  
  const response = await axios.post(`${API_BASE_URL}/load_model`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const searchImage = async (formData) => {
  // formData should contain 'image' (file) or 'image_path' and 'model'
  const response = await axios.post(`${API_BASE_URL}/search`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const analyzeSimilarity = async (queryPath, resultPath, model) => {
  const formData = new FormData();
  formData.append('query_path', queryPath);
  formData.append('result_path', resultPath);
  formData.append('model', model);
  
  const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getImageUrl = (path) => {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return `${API_BASE_URL}/images/${cleanPath}`;
};
