import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { login as apiLogin, registerAndLogin as apiRegisterAndLogin } from '../api/auth';
import { setLogoutHandler, clearToken } from '../api/client';
import type { LoginResponse } from '../types';

interface AuthState {
  user: LoginResponse | null;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<LoginResponse | null>(() => {
    const raw = localStorage.getItem('auth_user');
    if (raw) {
      try { return JSON.parse(raw); } catch { /* ignore */ }
    }
    return null;
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const logout = useCallback(() => {
    clearToken();
    localStorage.removeItem('auth_user');
    setUser(null);
    setError(null);
  }, []);

  useEffect(() => {
    setLogoutHandler(logout);
  }, [logout]);

  const login = useCallback(async (username: string, password: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiLogin(username, password);
      localStorage.setItem('auth_user', JSON.stringify(result));
      setUser(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Login failed');
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiRegisterAndLogin(username, password);
      localStorage.setItem('auth_user', JSON.stringify(result));
      setUser(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Registration failed');
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, isLoading, error, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
