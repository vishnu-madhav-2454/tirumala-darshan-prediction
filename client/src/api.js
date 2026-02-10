/**
 * API service — communicates with the Flask backend
 * In production: same origin (Flask serves React build)
 * In development: proxy to localhost:5000
 */
import axios from "axios";

const isDev = import.meta.env.DEV;

const API = axios.create({
  baseURL: isDev ? "http://localhost:5000/api" : "/api",
  timeout: 120_000,
});

export const getHealth = () => API.get("/health");

export const predictToday = () => API.get("/predict/today");

export const predictDays = (days = 7) => API.get(`/predict?days=${days}`);

export const predictDate = (date) =>
  API.post("/predict/date", { date });

export const predictRange = (start, end) =>
  API.get(`/predict/range?start=${start}&end=${end}`);

export const getDataSummary = () => API.get("/data/summary");

export const getHistory = (page = 1, perPage = 50, year, month) => {
  const params = new URLSearchParams({ page, per_page: perPage });
  if (year) params.append("year", year);
  if (month) params.append("month", month);
  return API.get(`/data/history?${params}`);
};

export default API;
