import axios from 'axios'
import { refreshWithoutRepeats } from './api'

// const instance = axios.create({ baseURL: 'http://127.0.0.1:8000' })
const instance = axios.create({
  baseURL: '/',
  withCredentials: true,
  responseType: 'json',
})

/* ==$API with  response interceptors== */

instance.interceptors.response.use(
  (config) => config,
  async (error) => {
    const originalRequest = error.config
    if (
      error.response.status === 401 &&
      originalRequest &&
      !originalRequest._isRetry
    ) {
      originalRequest._isRetry = true
      try {
        await refreshWithoutRepeats()

        return await instance.request(originalRequest)
      } catch (e) {
        // eslint-disable-next-line
        console.log(e)
      }
    }
    throw error
  }
)
export default instance
