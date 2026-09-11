"use client";

import { createContext, useContext, useEffect, useMemo } from "react";
import type { UserProfile } from "./types";

const TOKEN_COOKIE = "qknee_token";
const USER_STORAGE_KEY = "qknee_user";

function clearCookie(name: string) {
  if (typeof document !== "undefined") {
    document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
  }
}

interface AuthContextValue {
  token: string | null;
  user: UserProfile | null;
  isReady: boolean;
  signIn: (token: string, user: UserProfile) => void;
  signOut: () => void;
}

const defaultAuthValue: AuthContextValue = {
  token: null,
  user: null,
  isReady: true,
  signIn: () => {},
  signOut: () => {},
};

const AuthContext = createContext<AuthContextValue>(defaultAuthValue);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Purge any legacy authentication state from prior builds
    try {
      clearCookie(TOKEN_COOKIE);
      if (typeof window !== "undefined") {
        window.localStorage.removeItem(USER_STORAGE_KEY);
      }
    } catch {
      // Storage unavailable or restricted
    }
  }, []);

  const value = useMemo(() => defaultAuthValue, []);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  return useContext(AuthContext) ?? defaultAuthValue;
}
