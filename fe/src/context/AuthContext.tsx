import { createContext, useContext, useState, useEffect } from "react";
import { jwtDecode } from "jwt-decode";

interface AuthUser {
  id: number;
  role: string;
}

interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  setToken: (token: string | null) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [token, setTokenState] = useState<string | null>(localStorage.getItem("token"));
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    if (token) {
      try {
        const decoded = jwtDecode<{ sub: string; role: string }>(token);
        setUser({ id: Number(decoded.sub), role: decoded.role });
      } catch {
        setUser(null);
      }
    } else {
      setUser(null);
    }
  }, [token]);

  const setToken = (t: string | null) => {
    if (t) localStorage.setItem("token", t);
    else localStorage.removeItem("token");
    setTokenState(t);
  };

  const logout = () => setToken(null);

  return (
    <AuthContext.Provider value={{ user, token, setToken, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};
