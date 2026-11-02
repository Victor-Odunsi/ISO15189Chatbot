export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-error/30 bg-error-bg px-3 py-2 text-sm text-error">
      {message}
    </div>
  );
}
