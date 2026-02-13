import axios from "axios";

const isDev = import.meta.env.DEV;

const API = axios.create({
  baseURL: isDev ? "http://localhost:5000/api" : "/api",
  timeout: 120_000,
});

export const getHealth = () => API.get("/health");
export const predictToday = () => API.get("/predict/today");
export const predictDays = (days = 7) => API.get(`/predict?days=${days}`);
export const predictRange = (start, end) => API.post("/predict", { start_date: start, end_date: end });
export const getDataSummary = () => API.get("/data/summary");
export const getModelInfo = () => API.get("/model-info");

export const getHistory = (page = 1, perPage = 50, startDate, endDate) => {
  const params = new URLSearchParams({ page, per_page: perPage });
  if (startDate) params.append("start_date", startDate);
  if (endDate) params.append("end_date", endDate);
  return API.get(`/history?${params}`);
};

export const sendChatMessage = (message) => API.post("/chat", { message });
export const getTripPlan = (params) => API.post("/trip/plan", params);
export const getCalendar = (year, month) => API.get(`/calendar/${year}/${month}`);

export default API;
