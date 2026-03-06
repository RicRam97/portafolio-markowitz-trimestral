import AuthContainer from '@/components/auth/AuthContainer';
import { Suspense } from 'react';

export default function LoginPage() {
    return (
        <div className="auth-body">
            {/* Background orbs (matching landing page aesthetic) */}
            <div className="auth-bg-orbs">
                <div className="hero-orb hero-orb--blue"></div>
                <div className="hero-orb hero-orb--green"></div>
                <div className="hero-orb hero-orb--purple"></div>
            </div>

            <Suspense fallback={<div className="glass-panel" style={{ width: 400, height: 500, margin: 'auto' }}></div>}>
                <AuthContainer />
            </Suspense>
        </div>
    );
}
