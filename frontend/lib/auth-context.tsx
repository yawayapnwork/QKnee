"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { UserProfile } from "./types";

export type AuthMode = "guest" | "authenticated";

const TOKEN_COOKIE = "qknee_token";
const USER_STORAGE_KEY = "qknee_user";
const AUTH_MODE_KEY = "qknee_auth_mode";

function setCookie(name: string, value: string, days: number) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
}

function getCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function clearCookie(name: string) {
  document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
}

interface AuthContextValue {
  token: string | null;
  user: UserProfile | null;
  authMode: AuthMode;
  isReady: boolean;
  signIn: (token: string, user: UserProfile) => void;
  signOut: () => void;
  continueAsGuest: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("guest");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const storedToken = getCookie(TOKEN_COOKIE);
    const storedUser = window.localStorage.getItem(USER_STORAGE_KEY);
    const storedMode = window.localStorage.getItem(AUTH_MODE_KEY) as AuthMode | null;

    // Purge deprecated mock session token if present from prior builds
    if (storedToken === "institutional-demo-session-token") {
      clearCookie(TOKEN_COOKIE);
      window.localStorage.removeItem(USER_STORAGE_KEY);
      window.localStorage.setItem(AUTH_MODE_KEY, "guest");
      setAuthMode("guest");
      setIsReady(true);
      return;
    }

    if (storedToken && storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser) as UserProfile;
        setToken(storedToken);
        setUser(parsedUser);
        setAuthMode("authenticated");
      } catch {
        clearCookie(TOKEN_COOKIE);
        window.localStorage.removeItem(USER_STORAGE_KEY);
        window.localStorage.setItem(AUTH_MODE_KEY, "guest");
        setAuthMode("guest");
      }
    } else {
      setAuthMode(storedMode === "authenticated" && storedToken ? "authenticated" : "guest");
    }
    setIsReady(true);
  }, []);

  const signIn = useCallback((newToken: string, newUser: UserProfile) => {
    setCookie(TOKEN_COOKIE, newToken, 1);
    window.localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(newUser));
    window.localStorage.setItem(AUTH_MODE_KEY, "authenticated");
    setToken(newToken);
    setUser(newUser);
    setAuthMode("authenticated");
  }, []);

  const signOut = useCallback(() => {
    clearCookie(TOKEN_COOKIE);
    window.localStorage.removeItem(USER_STORAGE_KEY);
    window.localStorage.setItem(AUTH_MODE_KEY, "guest");
    setToken(null);
    setUser(null);
    setAuthMode("guest");
  }, []);

  const continueAsGuest = useCallback(() => {
    clearCookie(TOKEN_COOKIE);
    window.localStorage.removeItem(USER_STORAGE_KEY);
    window.localStorage.setItem(AUTH_MODE_KEY, "guest");
    setToken(null);
    setUser(null);
    setAuthMode("guest");
  }, []);

  const value = useMemo(
    () => ({ token, user, authMode, isReady, signIn, signOut, continueAsGuest }),
    [token, user, authMode, isReady, signIn, signOut, continueAsGuest],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
