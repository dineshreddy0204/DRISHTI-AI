import { baseUrl } from "@/api/baseUrl";
import { cn } from "@/lib/utils";

/** Original DRISHTI AI realistic emblem logo. */
type LogoProps = { className?: string };
export default function Logo({ className }: LogoProps) {
  return (
    <img
      src={`${baseUrl}images/drishti-logo.jpg`}
      alt="DRISHTI AI"
      className={cn("rounded-lg object-cover shadow-sm", className)}
    />
  );
}
