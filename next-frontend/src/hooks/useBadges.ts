'use client';

import { createBrowserClient } from '@supabase/ssr';
import { toast } from 'sonner';
import { BADGES } from '@/lib/badges';

function getSupabase() {
    return createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    );
}

export async function awardBadge(userId: string, badgeId: string) {
    const badge = BADGES[badgeId];
    if (!badge) return;

    const supabase = getSupabase();

    // Check if user already has this badge
    const { count } = await supabase
        .from('user_badges')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', userId)
        .eq('badge_id', badgeId);

    if (count && count > 0) return; // Already earned

    const { error } = await supabase
        .from('user_badges')
        .insert({ user_id: userId, badge_id: badgeId });

    if (error) return;

    toast.success(`${badge.icon} Nuevo logro: ${badge.name}`, {
        description: badge.description,
        duration: 5000,
    });
}

export async function checkAndAwardBadges(
    userId: string,
    context: {
        sharpeRatio?: number;
        tickerCount?: number;
        strategyCount?: number;
    },
) {
    const promises: Promise<void>[] = [];

    // First strategy
    if (context.strategyCount !== undefined && context.strategyCount >= 1) {
        promises.push(awardBadge(userId, 'first_strategy'));
    }

    // 5 strategies
    if (context.strategyCount !== undefined && context.strategyCount >= 5) {
        promises.push(awardBadge(userId, 'five_strategies'));
    }

    // Sharpe > 1.5
    if (context.sharpeRatio !== undefined && context.sharpeRatio > 1.5) {
        promises.push(awardBadge(userId, 'sharpe_master'));
    }

    // Diversifier: 5+ tickers
    if (context.tickerCount !== undefined && context.tickerCount >= 5) {
        promises.push(awardBadge(userId, 'diversifier'));
    }

    await Promise.allSettled(promises);
}
