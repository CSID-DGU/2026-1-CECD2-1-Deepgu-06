import { createContext, useContext, useState } from "react";
import { jwtDecode } from "jwt-decode";

interface AuthUser {
  id: number;
  role: "ADMIN" | "USER";
}

interface AuthSession {
  token: string;
  user: AuthUser;
}

interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  setToken: (token: string | null) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

const decodeUser = (token: string): AuthUser | null => {
  try {
    const decoded = jwtDecode<{ sub: string; role: string; exp?: number }>(token);
    const id = Number(decoded.sub);

    if (!Number.isInteger(id) || !["ADMIN", "USER"].includes(decoded.role)) {
      return null;
    }
    if (decoded.exp !== undefined && decoded.exp * 1000 <= Date.now()) {
      return null;
    }

    return { id, role: decoded.role as AuthUser["role"] };
  } catch {
    return null;
  }
};

const readStoredSession = (): AuthSession | null => {
  const token = localStorage.getItem("token");
  if (!token) return null;

  const user = decodeUser(token);
  if (!user) {
    localStorage.removeItem("token");
    return null;
  }
  return { token, user };
};

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [session, setSession] = useState<AuthSession | null>(() => readStoredSession());

  const setToken = (t: string | null) => {
    if (!t) {
      localStorage.removeItem("token");
      setSession(null);
      return;
    }

    const user = decodeUser(t);
    if (!user) {
      localStorage.removeItem("token");
      setSession(null);
      return;
    }

    localStorage.setItem("token", t);
    setSession({ token: t, user });
  };

  const logout = () => setToken(null);

  return (
    <AuthContext.Provider value={{
      user: session?.user ?? null,
      token: session?.token ?? null,
      setToken,
      logout,
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};
