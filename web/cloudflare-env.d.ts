declare namespace Cloudflare {
  interface Env {
    CAREER_API_BASE_URL?: string;
    CAREER_SERVICE_TOKEN?: string;
    CAREER_ALLOW_INSECURE_LOOPBACK?: string;
    DB?: D1Database;
    BUCKET?: R2Bucket;
  }
}
