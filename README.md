# astro-tailwind-starter

![Astro](https://img.shields.io/badge/Astro-0C1222?style=for-the-badge&logo=astro&logoColor=FDFDFE)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Cloudflare](https://img.shields.io/badge/Cloudflare-F38020?style=for-the-badge&logo=Cloudflare&logoColor=white)
![Prettier](https://img.shields.io/badge/prettier-1A2C34?style=for-the-badge&logo=prettier&logoColor=F7BA3E)
![ESLint](https://img.shields.io/badge/eslint-3A33D1?style=for-the-badge&logo=eslint&logoColor=white)
![MIT](https://img.shields.io/badge/MIT-green?style=for-the-badge)

## Minimum Setup for Astro Development

This is a starter kit pre-configured with Astro + Tailwind CSS + Prettier + ESLint. It eliminates the initial environment setup, allowing you to focus immediately on design and development.

[![Use this template](https://img.shields.io/badge/use_this_template-197935?style=for-the-badge&logo=github&logoColor=white)](https://github.com/new?template_name=astro-tailwind-starter&template_owner=ryotahagihara)
[![View on GitHub](https://img.shields.io/badge/view_on_github-1f2328?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ryotahagihara/astro-tailwind-starter)

## Try Online

You can edit and preview the code instantly in your browser via the buttons below.

[![Open in StackBlitz](https://developer.stackblitz.com/img/open_in_stackblitz.svg)](https://stackblitz.com/github/ryotahagihara/astro-tailwind-starter)
[![Edit in CodeSandbox](https://codesandbox.io/static/img/play-codesandbox.svg)](https://codesandbox.io/s/github/ryotahagihara/astro-tailwind-starter)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ryotahagihara/astro-tailwind-starter)

## Tech Stack

| Tech               | Description                                     |
| ------------------ | ----------------------------------------------- |
| Astro v5           | The web framework for content-driven websites   |
| TypeScript         | JavaScript with syntax for types                |
| Tailwind CSS v4    | A utility-first CSS framework                   |
| Cloudflare Workers | Serverless platform running on a global network |
| Prettier           | An opinionated code formatter                   |
| ESLint             | A pluggable linting utility for JavaScript      |

## Features

### Style-Agnostic

Contains no specific themes. However, Tailwind Typography is included for styled prose content, ensuring basic default styling without extra configuration.

### Clean & Accessible

Pre-configured with Prettier and ESLint. Automatically maintains code quality and accessibility standards.

### Standard Configuration

Uses mostly default settings. No complex custom rules; follows the official documentation.

### Ready to Code

Includes only the essential configuration required to start development. Minimal setup time, ready to code.

## Getting Started

### 1. Use as Astro Template

Create a project directly from your terminal using the Astro CLI.

```bash
npm create astro@latest -- --template ryotahagihara/astro-tailwind-starter
```

### 2. Use as GitHub Template

Click the "Use this template" button on the GitHub repository page to create a new repository.

### 3. Deploy via Services

Click the "Deploy" buttons below to automatically create a repository and deploy.

## CLI Commands

### Install Dependencies

```bash
npm install
```

### Start Development Server

```bash
npm run dev
```

### Build Production Site

```bash
npm run build
```

### Preview Build Locally

```bash
npm run preview
```

## Project Structure

Follows Astro's official directory structure conventions.

```text
.
├── src/
│   ├── components/
│   │   ├── Header.astro    # Header component
│   │   └── Footer.astro    # Footer component
│   ├── layouts/
│   │   └── Layout.astro    # Base layout
│   ├── pages/
│   │   └── index.astro     # Top page
│   └── styles/
│       └── global.css      # Tailwind CSS settings
├── public/
│   └── favicon.svg         # Favicon
├── .prettierrc             # Prettier config
├── eslint.config.js        # ESLint config
├── astro.config.mjs        # Astro config
├── wrangler.jsonc          # Cloudflare Workers config
├── tsconfig.json           # TypeScript config
└── package.json            # Dependencies
```

## Deployment

Automatically create a repository and deploy to hosting services via the buttons below.

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/ryotahagihara/astro-tailwind-starter)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/ryotahagihara/astro-tailwind-starter)
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/ryotahagihara/astro-tailwind-starter)

### Cloudflare Workers

Configured for Cloudflare Workers. Includes necessary configuration for deployment via Wrangler.

```bash
npm run build && npx wrangler deploy
```

**Note:** Requires Cloudflare authentication on first run.

> Cloudflare recommends using Cloudflare Workers for new projects. See [Deploy your Astro Site to Cloudflare | Docs](https://docs.astro.build/en/guides/deploy/cloudflare/) for details.

### Other Environments

Builds as a pure static site (SSG), compatible with Vercel, Netlify, GitHub Pages, or any standard server. See [Deploy your Astro Site | Docs](https://docs.astro.build/en/guides/deploy/) for details.

## Resources

- [Astro](https://astro.build) - [Docs](https://docs.astro.build/en/getting-started/)
- [TypeScript](https://www.typescriptlang.org) - [Docs](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com) - [Docs](https://tailwindcss.com/docs/installation/using-vite)
- [Cloudflare Workers](https://www.cloudflare.com/developer-platform/products/workers/) - [Docs](https://developers.cloudflare.com/workers/)
- [Prettier](https://prettier.io) - [Docs](https://prettier.io/docs/)
- [ESLint](https://eslint.org) - [Docs](https://eslint.org/docs/latest/)

## Contact

Please report issues or feature requests via [GitHub Issues](https://github.com/ryotahagihara/astro-tailwind-starter/issues).

## License

MIT License - See [LICENSE](LICENSE) for details.

Copyright &copy; 2025 [Ryota Hagihara](https://www.ryotahagihara.com/) All rights reserved.
