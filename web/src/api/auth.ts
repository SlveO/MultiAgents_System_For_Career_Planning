import { apiPost } from './client';
import { setToken } from './client';
import type { LoginResponse, RegisterRequest, LoginRequest, UserRow } from '../types';

export async function login(
  username: string,
  password: string
): Promise<LoginResponse> {
  const body: LoginRequest = { username, password };
  const res = await apiPost('/auth/login', body);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(err.detail || 'Login failed');
  }
  const data: LoginResponse = await res.json();
  setToken(data.access_token);
  return data;
}

export async function register(
  username: string,
  password: string
): Promise<UserRow> {
  const body: RegisterRequest = { username, password };
  const res = await apiPost('/auth/register', body);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Registration failed' }));
    throw new Error(err.detail || 'Registration failed');
  }
  return res.json();
}

export async function registerAndLogin(
  username: string,
  password: string
): Promise<LoginResponse> {
  await register(username, password);
  return login(username, password);
}
