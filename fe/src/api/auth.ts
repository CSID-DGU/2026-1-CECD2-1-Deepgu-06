import client from "./client";

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export const login = async (data: LoginRequest): Promise<string> => {
  const res = await client.post("/api/auth/login", data);
  return res.data.data.access_token;
};

export const register = async (data: RegisterRequest) => {
  const res = await client.post("/api/auth/register", data);
  return res.data.data;
};
