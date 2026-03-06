'use client';

interface SkeletonCardProps {
    lines?: number;
    className?: string;
}

export default function SkeletonCard({ lines = 3, className = '' }: SkeletonCardProps) {
    return (
        <div className={`glass-panel p-5 ${className}`}>
            {Array.from({ length: lines }).map((_, i) => (
                <div
                    key={i}
                    className="skeleton mb-3 last:mb-0"
                    style={{
                        height: i === 0 ? '20px' : '14px',
                        width: i === 0 ? '60%' : i === lines - 1 ? '40%' : '80%',
                    }}
                />
            ))}
        </div>
    );
}
