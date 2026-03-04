# Configuración de Google OAuth para Kaudal

## 1. Crear proyecto en Google Cloud Console

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Anota el **Project ID**

## 2. Configurar OAuth Consent Screen

1. En el menú lateral: **APIs & Services** → **OAuth consent screen**
2. Selecciona **External** → Click **Create**
3. Llena los campos:
   - **App name**: `Kaudal`
   - **User support email**: tu email
   - **Developer contact**: tu email
4. **Scopes**: Agrega `email` y `profile`
5. **Test users**: Agrega tu email mientras estés en modo testing
6. Click **Save and Continue** hasta terminar

## 3. Crear OAuth 2.0 Client ID

1. Ve a **APIs & Services** → **Credentials**
2. Click **+ CREATE CREDENTIALS** → **OAuth client ID**
3. **Application type**: `Web application`
4. **Name**: `Kaudal Web`
5. **Authorized redirect URIs**: Agrega exactamente:
   ```
   https://aqjfojjybusgfcqqlcwl.supabase.co/auth/v1/callback
   ```
6. Click **Create**
7. Copia el **Client ID** y **Client Secret**

## 4. Configurar en Supabase

1. Ve a tu [Supabase Dashboard](https://supabase.com/dashboard)
2. Selecciona tu proyecto
3. Ve a **Authentication** → **Providers**
4. Busca **Google** y actívalo (toggle ON)
5. Pega:
   - **Client ID**: el que copiaste de Google
   - **Client Secret**: el que copiaste de Google
6. Click **Save**

## 5. Ejecutar migración SQL

En **SQL Editor** de Supabase, ejecuta:

```sql
-- Agregar columna onboarding_complete a profiles (si no existe)
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS onboarding_complete BOOLEAN NOT NULL DEFAULT false;
```

## 6. Verificar

1. Abre `http://localhost:5173/login.html`
2. Click "Continuar con Google"
3. Deberías ver el popup de Google para seleccionar cuenta
4. Al completar, serás redirigido al dashboard o al Test de Sueños

> **Nota**: En modo "Testing" de Google, solo los emails agregados como test users podrán autenticarse. Para producción, necesitas publicar la app (Submit for verification).
