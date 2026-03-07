import { toast } from 'sonner';

export function useNotification() {
    return {
        success: (mensaje: string) =>
            toast.success(mensaje, {
                duration: 4000,
                closeButton: true,
            }),
        error: (mensaje: string) =>
            toast.error(mensaje, {
                duration: 4000,
                closeButton: true,
            }),
        warning: (mensaje: string) =>
            toast.warning(mensaje, {
                duration: 4000,
                closeButton: true,
            }),
        info: (mensaje: string) =>
            toast.info(mensaje, {
                duration: 4000,
                closeButton: true,
            }),
    };
}
