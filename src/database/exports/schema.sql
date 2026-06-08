--
-- PostgreSQL database dump
--

\restrict c7pP6PHJh5b3hk5CuaGoIn3e5eHhMgJpE9dfh6fRSwtVi2Kz8jKQAdrs9Z7qgg3

-- Dumped from database version 15.14 (Debian 15.14-1.pgdg13+1)
-- Dumped by pg_dump version 15.14 (Debian 15.14-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: definers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.definers (
    agent_id uuid NOT NULL,
    run_id uuid NOT NULL,
    task_id text NOT NULL,
    agent text NOT NULL,
    parent_explorer_ids uuid[] DEFAULT '{}'::uuid[] NOT NULL,
    output jsonb,
    metadata jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: evals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evals (
    run_id uuid NOT NULL,
    system text NOT NULL,
    dataset text NOT NULL,
    data jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: explorers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.explorers (
    agent_id uuid NOT NULL,
    run_id uuid NOT NULL,
    task_id text NOT NULL,
    agent text NOT NULL,
    output jsonb,
    metadata jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version text NOT NULL,
    applied_at timestamp with time zone DEFAULT now()
);


--
-- Name: definers definers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.definers
    ADD CONSTRAINT definers_pkey PRIMARY KEY (agent_id);


--
-- Name: definers definers_run_id_task_id_agent_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.definers
    ADD CONSTRAINT definers_run_id_task_id_agent_key UNIQUE (run_id, task_id, agent);


--
-- Name: evals evals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evals
    ADD CONSTRAINT evals_pkey PRIMARY KEY (run_id);


--
-- Name: explorers explorers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.explorers
    ADD CONSTRAINT explorers_pkey PRIMARY KEY (agent_id);


--
-- Name: explorers explorers_run_id_task_id_agent_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.explorers
    ADD CONSTRAINT explorers_run_id_task_id_agent_key UNIQUE (run_id, task_id, agent);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: definers_parent_explorers_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX definers_parent_explorers_idx ON public.definers USING gin (parent_explorer_ids);


--
-- Name: definers_run_task_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX definers_run_task_idx ON public.definers USING btree (run_id, task_id);


--
-- Name: explorers_run_task_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX explorers_run_task_idx ON public.explorers USING btree (run_id, task_id);


--
-- Name: definers definers_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.definers
    ADD CONSTRAINT definers_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.evals(run_id) ON DELETE CASCADE;


--
-- Name: explorers explorers_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.explorers
    ADD CONSTRAINT explorers_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.evals(run_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict c7pP6PHJh5b3hk5CuaGoIn3e5eHhMgJpE9dfh6fRSwtVi2Kz8jKQAdrs9Z7qgg3

