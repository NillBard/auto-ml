import axios, { AxiosResponse } from 'axios';
import { instance } from '@/shared/libs';

export function login(email: string, password: string) {
    return axios.post(
        '/api/user',
        {
            login: email,
            password,
        },
        { withCredentials: true }
    )
}

export function logout() {
    return instance.delete('/api/user', { withCredentials: true })
}

export function register(email: string, password: string) {
    return axios.post('/api/user/register', {
        login: email,
        password,
    })
}

export interface RefreshResponse {
    refresh: string
}

export type TPromiseRefresh = Promise<AxiosResponse<RefreshResponse>>

export function refresh(token: string) {
    return axios.post<RefreshResponse>(
        '/api/user/refresh',
        { refresh: token },
        {
            withCredentials: true,
        }
    )
}

// Руками не трогать может взорваться!!!

let refreshPromise: TPromiseRefresh | null

export async function refreshWithoutRepeats() {
    const localCopy = refreshPromise
    let response: AxiosResponse<RefreshResponse>
    if (localCopy && refreshPromise) {
        response = await refreshPromise
    } else {
        refreshPromise = refresh(localStorage.getItem('refresh') || '')
        const copy: TPromiseRefresh = refreshPromise
        response = await copy
        refreshPromise = null
    }

    if (response.data && response.data.refresh) {
        localStorage.setItem('refresh', response.data.refresh)
    } else {
        localStorage.removeItem('refresh')
    }
}