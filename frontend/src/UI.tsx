import type { ReactNode } from 'react';
import { CircleAlert } from 'lucide-react';

export function PageHeading({ eyebrow, title, description, children }: {
  eyebrow: string; title: string; description: string; children?: ReactNode;
}) {
  return <div className="page-heading">
    <div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="lead">{description}</p></div>{children}
  </div>;
}

export function Notice({ children, danger = false }: { children: ReactNode; danger?: boolean }) {
  return <div className={`notice ${danger ? 'danger' : ''}`} role={danger ? 'alert' : undefined}><CircleAlert size={18}/><div>{children}</div></div>;
}

